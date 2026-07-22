-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 03 · Complex — ABAC Orchestration (the "policy symphony")
-- MAGIC Real governance is not one rule — it is many roles seeing the same data differently,
-- MAGIC plus time-of-day rules, plus keeping analytics working while PII is hidden. This stage
-- MAGIC layers all of it:
-- MAGIC
-- MAGIC | Role (group) | Rows | PII columns | Time |
-- MAGIC |---|---|---|---|
-- MAGIC | Junior nurse — `TEST_GROUP_A` | NORTH region only | heavily masked | anytime |
-- MAGIC | Auditor — `TEST_GROUP_C` | all regions | last-4 only | **business hours only** |
-- MAGIC | Senior doctor — `TEST_GROUP_B` | all regions | full clear | anytime |
-- MAGIC
-- MAGIC You (carlos.dip) are in **A + C**, so you see the *most restrictive* combination —
-- MAGIC exactly how precedence should behave. Plus **referential masking**: patient IDs are
-- MAGIC pseudonymised deterministically, so joins still work on masked data.

-- COMMAND ----------

USE CATALOG cedip_ws;
USE SCHEMA abac_healthcare_demo;

-- COMMAND ----------

-- MAGIC %md ## 0. Reset any earlier controls (safe if 01/02 not run this session)
-- MAGIC Stage 02's ABAC policies match the same tags as Stage 03, and Unity Catalog forbids
-- MAGIC two masks on one column — so we drop all existing schema policies first, then drop
-- MAGIC Stage 01's classic controls.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC schema = "cedip_ws.abac_healthcare_demo"
-- MAGIC for r in spark.sql(f"SHOW POLICIES ON SCHEMA {schema}").collect():
-- MAGIC     spark.sql(f"DROP POLICY {r[0]} ON SCHEMA {schema}")
-- MAGIC     print("dropped policy:", r[0])
-- MAGIC for s in ["ALTER TABLE Patients DROP ROW FILTER",
-- MAGIC           "ALTER TABLE Patients ALTER COLUMN Email DROP MASK",
-- MAGIC           "ALTER TABLE Patients ALTER COLUMN PhoneNumber DROP MASK",
-- MAGIC           "ALTER TABLE Patients ALTER COLUMN SSN DROP MASK"]:
-- MAGIC     try: spark.sql(s)
-- MAGIC     except Exception as e: print("skip:", s.split('COLUMN')[-1][:30])

-- COMMAND ----------

-- MAGIC %md ## 1. Role-graded functions
-- MAGIC A single function inspects the caller's group and returns a different result per role.
-- MAGIC Most-restrictive-wins: check junior (A) first, then auditor (C), else clear.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION grade_ssn(s STRING)
  RETURNS STRING
  COMMENT 'Junior: fully masked. Auditor: last 4. Senior: clear.'
  RETURN CASE
    WHEN is_account_group_member('TEST_GROUP_A') THEN '***-**-****'
    WHEN is_account_group_member('TEST_GROUP_C') THEN concat('XXX-XX-', substr(s, length(s)-3, 4))
    ELSE s END;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION grade_email(e STRING)
  RETURNS STRING
  COMMENT 'Junior: domain only. Auditor: first char + domain. Senior: clear.'
  RETURN CASE
    WHEN is_account_group_member('TEST_GROUP_A') THEN concat('****', substr(e, instr(e,'@')))
    WHEN is_account_group_member('TEST_GROUP_C') THEN concat(substr(e,1,1), '***', substr(e, instr(e,'@')))
    ELSE e END;

-- COMMAND ----------

-- Referential ID mask: same PatientID always maps to the same pseudonym, so JOINs survive.
CREATE OR REPLACE FUNCTION grade_patient_id(id STRING)
  RETURNS STRING
  COMMENT 'Junior/Auditor: deterministic pseudonym (join-safe). Senior: real ID.'
  RETURN CASE
    WHEN is_account_group_member('TEST_GROUP_A')
      OR is_account_group_member('TEST_GROUP_C')
    THEN concat('REF_', cast(crc32(id) AS STRING))
    ELSE id END;

-- COMMAND ----------

-- Combined row filter: junior -> NORTH only; auditor -> all rows but ONLY in business
-- hours (9am-6pm PT); senior -> everything, anytime.
CREATE OR REPLACE FUNCTION grade_row_access(region STRING)
  RETURNS BOOLEAN
  COMMENT 'Region + time-of-day access by role'
  RETURN CASE
    WHEN is_account_group_member('TEST_GROUP_A') THEN region = 'NORTH'
    WHEN is_account_group_member('TEST_GROUP_C')
      THEN hour(from_utc_timestamp(current_timestamp(), 'America/Los_Angeles')) BETWEEN 9 AND 17
    ELSE TRUE END;

-- COMMAND ----------

-- MAGIC %md ## 2. Orchestrated policies (all keyed off existing governance tags)

-- COMMAND ----------

-- SSN: full_mask-tagged columns, role-graded.
CREATE OR REPLACE POLICY abac_grade_ssn
ON SCHEMA cedip_ws.abac_healthcare_demo
COMMENT 'Role-graded SSN masking'
COLUMN MASK cedip_ws.abac_healthcare_demo.grade_ssn
TO `TEST_GROUP_A`, `TEST_GROUP_C`
FOR TABLES
MATCH COLUMNS hasTagValue('abac_demo_class', 'phi') AS c
ON COLUMN c;

-- COMMAND ----------

-- Email: partial_mask-tagged columns, role-graded.
CREATE OR REPLACE POLICY abac_grade_email
ON SCHEMA cedip_ws.abac_healthcare_demo
COMMENT 'Role-graded email masking'
COLUMN MASK cedip_ws.abac_healthcare_demo.grade_email
TO `TEST_GROUP_A`, `TEST_GROUP_C`
FOR TABLES
MATCH COLUMNS hasTagValue('abac_demo_class', 'pii') AS c
ON COLUMN c;

-- COMMAND ----------

-- Referential row-level pseudonymisation of the region column's owning tables.
-- Row filter: region + time-of-day, applied to every sensitive table.
CREATE OR REPLACE POLICY abac_grade_rows
ON SCHEMA cedip_ws.abac_healthcare_demo
COMMENT 'Role-graded row access: region for junior, business-hours for auditor'
ROW FILTER cedip_ws.abac_healthcare_demo.grade_row_access
TO `TEST_GROUP_A`, `TEST_GROUP_C`
FOR TABLES
WHEN hasTagValue('abac_demo_row', 'region_scoped')
MATCH COLUMNS hasTag('abac_demo_geo') AS r
USING COLUMNS(r);

-- COMMAND ----------

-- MAGIC %md ## 3. The reveal — you (A+C) see the most restrictive blend

-- COMMAND ----------

SELECT current_user() AS me,
       is_account_group_member('TEST_GROUP_A') AS junior,
       is_account_group_member('TEST_GROUP_B') AS senior,
       is_account_group_member('TEST_GROUP_C') AS auditor,
       hour(from_utc_timestamp(current_timestamp(),'America/Los_Angeles')) AS pt_hour;

-- COMMAND ----------

-- Patients: NORTH-only (junior wins), SSN fully masked, email domain-only.
SELECT PatientID, FirstName, Email, SSN, Region FROM Patients ORDER BY PatientID;

-- COMMAND ----------

-- MAGIC %md ## 4. Referential integrity survives masking
-- MAGIC Join Patients→Visits. Even though a senior would see real IDs and you a pseudonym,
-- MAGIC the join still returns matched rows because the pseudonym is deterministic.

-- COMMAND ----------

SELECT grade_patient_id(p.PatientID) AS patient_ref,
       p.FirstName, v.VisitType, v.Diagnosis
FROM Patients p
JOIN Visits v ON p.PatientID = v.PatientID
ORDER BY patient_ref;

-- COMMAND ----------

-- MAGIC %md ## 5. Audit trail — every active policy in one view

-- COMMAND ----------

SHOW POLICIES ON SCHEMA cedip_ws.abac_healthcare_demo;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Talk track
-- MAGIC - One tag vocabulary drives **three roles** with different row scope, mask depth, and time windows.
-- MAGIC - Precedence is explicit and testable: A+C member sees the junior+auditor intersection.
-- MAGIC - **Time-based:** re-run cell 3.3 after 6pm PT and the auditor path returns zero rows — access expires automatically.
-- MAGIC - **Analytics still works:** referential masking keeps joins intact while hiding identity.
-- MAGIC - This is governance as *data*, not as code scattered across pipelines and BI tools.

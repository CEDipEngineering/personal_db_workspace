-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 02 · Medium — ABAC: Tag Once, Govern Everywhere
-- MAGIC Stage 01 attached a mask to **each column, on each table, by hand**. That does not scale
-- MAGIC to thousands of tables. **Attribute-Based Access Control (ABAC)** flips it: you *tag* data
-- MAGIC with its sensitivity, write the policy **once at the schema level**, and Unity Catalog
-- MAGIC applies it to every column/table carrying that tag — including tables created *later*.
-- MAGIC
-- MAGIC Tags were applied in `00_setup` using governed keys (`abac_demo_class`, `abac_demo_row`,
-- MAGIC `abac_demo_geo`). Here we write the policies that consume them.
-- MAGIC
-- MAGIC Personas: policies target `TEST_GROUP_A` (you) → you see masked/filtered.
-- MAGIC `TEST_GROUP_B` (not you) sees everything.

-- COMMAND ----------

USE CATALOG cedip_ws;
USE SCHEMA abac_healthcare_demo;

-- COMMAND ----------

-- MAGIC %md ## 0. Reset — drop Stage 01's manual controls + any prior ABAC policies
-- MAGIC So the contrast is purely this stage's ABAC. (Safe if nothing was run before.)

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
-- MAGIC     except Exception: pass

-- COMMAND ----------

-- MAGIC %md ## 1. See the tags that drive everything

-- COMMAND ----------

SELECT table_name, column_name, tag_name, tag_value
FROM information_schema.column_tags
WHERE schema_name = 'abac_healthcare_demo'
ORDER BY tag_name, table_name, column_name;

-- COMMAND ----------

-- MAGIC %md ## 2. Column-mask policies — one policy per masking rule, applied by tag
-- MAGIC `MATCH COLUMNS hasTagValue('abac_demo_class', <class>) AS c ON COLUMN c` means: *for any
-- MAGIC column, on any table in this schema, tagged with this rule — mask it for `TEST_GROUP_A`.*

-- COMMAND ----------

-- Rule: exactly ONE mask policy per tag value, and every column carrying that value must
-- match the mask function's argument type. partial_mask columns (Email, PhoneNumber,
-- Providers.Email) are all STRING.
CREATE OR REPLACE POLICY abac_mask_partial
ON SCHEMA cedip_ws.abac_healthcare_demo
COMMENT 'ABAC: partially mask any STRING column tagged abac_demo_class=pii'
COLUMN MASK cedip_ws.abac_healthcare_demo.mask_string_partial
TO `TEST_GROUP_A`
FOR TABLES
MATCH COLUMNS hasTagValue('abac_demo_class', 'pii') AS c
ON COLUMN c;

-- COMMAND ----------

-- full_mask STRING columns: SSN, Insurance.PolicyNumber -> '***'
CREATE OR REPLACE POLICY abac_mask_full
ON SCHEMA cedip_ws.abac_healthcare_demo
COMMENT 'ABAC: fully mask any STRING column tagged abac_demo_class=phi (SSN, policy#)'
COLUMN MASK cedip_ws.abac_healthcare_demo.mask_full
TO `TEST_GROUP_A`
FOR TABLES
MATCH COLUMNS hasTagValue('abac_demo_class', 'phi') AS c
ON COLUMN c;

-- COMMAND ----------

-- redact STRING columns: Patients.Address -> '[REDACTED]'
CREATE OR REPLACE POLICY abac_mask_redact
ON SCHEMA cedip_ws.abac_healthcare_demo
COMMENT 'ABAC: redact any STRING column tagged abac_demo_class=redact (street address)'
COLUMN MASK cedip_ws.abac_healthcare_demo.mask_redact
TO `TEST_GROUP_A`
FOR TABLES
MATCH COLUMNS hasTagValue('abac_demo_class', 'redact') AS c
ON COLUMN c;

-- COMMAND ----------

-- DateOfBirth is DATE-typed (tagged abac_demo_class=dob) -> collapse to Jan 1 of birth year.
CREATE OR REPLACE POLICY abac_mask_dob
ON SCHEMA cedip_ws.abac_healthcare_demo
COMMENT 'ABAC: mask any DATE column tagged abac_demo_class=dob (DateOfBirth) to year only'
COLUMN MASK cedip_ws.abac_healthcare_demo.mask_dob_year
TO `TEST_GROUP_A`
FOR TABLES
MATCH COLUMNS hasTagValue('abac_demo_class', 'dob') AS c
ON COLUMN c;

-- COMMAND ----------

-- MAGIC %md ## 3. Row-filter policy — applied to every table tagged sensitive
-- MAGIC `WHEN hasTagValue('abac_demo_row','region_scoped')` selects the tables; `MATCH COLUMNS
-- MAGIC hasTag('abac_demo_geo') AS r ... USING COLUMNS(r)` feeds the region column
-- MAGIC to the filter function. `TEST_GROUP_A` is restricted to NORTH.

-- COMMAND ----------

CREATE OR REPLACE POLICY abac_row_region
ON SCHEMA cedip_ws.abac_healthcare_demo
COMMENT 'ABAC: TEST_GROUP_A sees only NORTH rows on any table tagged abac_demo_row=region_scoped'
ROW FILTER cedip_ws.abac_healthcare_demo.region_filter
TO `TEST_GROUP_A`
FOR TABLES
WHEN hasTagValue('abac_demo_row', 'region_scoped')
MATCH COLUMNS hasTag('abac_demo_geo') AS r
USING COLUMNS(r);

-- COMMAND ----------

-- MAGIC %md ## 4. The reveal — one policy set, applied across MULTIPLE tables at once

-- COMMAND ----------

-- Patients: masked columns + NORTH-only rows
SELECT PatientID, FirstName, Email, PhoneNumber, Address, SSN, DateOfBirth, Region
FROM Patients ORDER BY PatientID;

-- COMMAND ----------

-- Visits: SAME row policy auto-applied (tagged sensitive + has region column) — NORTH only.
-- No per-table work was done for Visits. That's the ABAC payoff.
SELECT VisitID, PatientID, VisitType, Diagnosis, Region FROM Visits ORDER BY VisitID;

-- COMMAND ----------

-- Insurance PolicyNumber (tagged full_mask) auto-masked too — a different table entirely.
SELECT InsuranceID, PatientID, InsuranceCompany, PolicyNumber FROM Insurance ORDER BY InsuranceID;

-- COMMAND ----------

-- MAGIC %md ## 5. Audit — list the active policies

-- COMMAND ----------

SHOW POLICIES ON SCHEMA cedip_ws.abac_healthcare_demo;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Talk track
-- MAGIC - Four policies now govern PII/PHI across **every** table in the schema — no per-column ALTERs.
-- MAGIC - Add a new table tomorrow with a `abac_demo_class`-tagged column → it is masked automatically.
-- MAGIC - Governance follows the *data attribute* (the tag), not the object. That is ABAC.
-- MAGIC - Next (03): overlapping policies for multiple roles + time-based access + join-safe masking.

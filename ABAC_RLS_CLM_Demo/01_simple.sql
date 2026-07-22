-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 01 · Simple — RLS + CLM Fundamentals
-- MAGIC **The pitch:** *same query, same table, same instant — different results based on who you are.*
-- MAGIC
-- MAGIC This stage uses the **classic** Unity Catalog controls, attached to one table:
-- MAGIC - **Row-Level Security (RLS):** a row filter so `TEST_GROUP_A` sees only NORTH-region patients.
-- MAGIC - **Column-Level Masking (CLM):** column masks so `TEST_GROUP_A` sees masked email / phone / SSN.
-- MAGIC
-- MAGIC You (carlos.dip) are in `TEST_GROUP_A`, so **you** experience the restriction directly.
-- MAGIC Members of `TEST_GROUP_B` (which you are not in) see everything — that is the live contrast.

-- COMMAND ----------

USE CATALOG cedip_ws;
USE SCHEMA abac_healthcare_demo;

-- COMMAND ----------

-- MAGIC %md ## 0. Reset — clear any policies from a prior stage so this notebook is re-runnable
-- MAGIC (No-op on a fresh setup. Ensures the baseline below truly shows unprotected data.)

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

-- MAGIC %md ## Baseline — before any policy (run first, note 12 rows, clear values)

-- COMMAND ----------

SELECT PatientID, FirstName, Email, PhoneNumber, SSN, Region
FROM Patients ORDER BY PatientID;

-- COMMAND ----------

-- MAGIC %md ## 1. Row-Level Security
-- MAGIC The row-filter function returns TRUE (keep row) unless the caller is in `TEST_GROUP_A`,
-- MAGIC in which case only NORTH rows survive. Attach it with `SET ROW FILTER`.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION rls_region_simple(region STRING)
  RETURNS BOOLEAN
  RETURN CASE WHEN is_account_group_member('TEST_GROUP_A') THEN region = 'NORTH'
              ELSE TRUE END;

-- COMMAND ----------

ALTER TABLE Patients SET ROW FILTER rls_region_simple ON (Region);

-- COMMAND ----------

-- MAGIC %md ## 2. Column-Level Masking
-- MAGIC Each mask function returns the clear value for everyone **except** `TEST_GROUP_A`,
-- MAGIC who sees the masked form. Attach with `SET MASK`.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION clm_email_simple(e STRING)
  RETURNS STRING
  RETURN CASE WHEN is_account_group_member('TEST_GROUP_A')
              THEN concat('****', substr(e, instr(e,'@')))
              ELSE e END;

CREATE OR REPLACE FUNCTION clm_phone_simple(p STRING)
  RETURNS STRING
  RETURN CASE WHEN is_account_group_member('TEST_GROUP_A')
              THEN concat('XXX-XXX-', substr(p, length(p)-3, 4))
              ELSE p END;

CREATE OR REPLACE FUNCTION clm_ssn_simple(s STRING)
  RETURNS STRING
  RETURN CASE WHEN is_account_group_member('TEST_GROUP_A')
              THEN concat('XXX-XX-', substr(s, length(s)-3, 4))
              ELSE s END;

-- COMMAND ----------

ALTER TABLE Patients ALTER COLUMN Email       SET MASK clm_email_simple;
ALTER TABLE Patients ALTER COLUMN PhoneNumber SET MASK clm_phone_simple;
ALTER TABLE Patients ALTER COLUMN SSN         SET MASK clm_ssn_simple;

-- COMMAND ----------

-- MAGIC %md ## 3. The reveal — run the SAME query again
-- MAGIC As a `TEST_GROUP_A` member (you): **6 NORTH rows**, email/phone/SSN masked.
-- MAGIC A `TEST_GROUP_B` member: **all 12 rows**, everything clear.

-- COMMAND ----------

SELECT PatientID, FirstName, Email, PhoneNumber, SSN, Region
FROM Patients ORDER BY PatientID;

-- COMMAND ----------

-- MAGIC %md ## 4. Prove it — who am I, and what did the filter do?

-- COMMAND ----------

SELECT current_user() AS me,
       is_account_group_member('TEST_GROUP_A') AS in_group_a,
       is_account_group_member('TEST_GROUP_B') AS in_group_b,
       count(*) AS rows_i_can_see
FROM Patients;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Talk track
-- MAGIC - No app logic changed. The governance lives in the table, enforced by the engine.
-- MAGIC - A junior analyst querying `Patients` is *physically unable* to read other regions or raw PII.
-- MAGIC - Next (02): instead of attaching masks column-by-column, we tag data once and let policy scale.

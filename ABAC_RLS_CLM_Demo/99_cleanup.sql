-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 99 · Cleanup — Full Teardown
-- MAGIC Removes **everything** the demo created: all ABAC policies, classic row filters /
-- MAGIC column masks, the governance tags, and finally the tables + schema.
-- MAGIC Run this to reset between demos or to fully decommission.
-- MAGIC
-- MAGIC The policy-drop step is a Python cell because `DROP POLICY` has no `IF EXISTS` — we
-- MAGIC enumerate live policies via `SHOW POLICIES` and drop whatever is there.

-- COMMAND ----------

USE CATALOG cedip_ws;
USE SCHEMA abac_healthcare_demo;

-- COMMAND ----------

-- MAGIC %md ## 1. Drop all ABAC policies on the schema (idempotent)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC schema = "cedip_ws.abac_healthcare_demo"
-- MAGIC rows = spark.sql(f"SHOW POLICIES ON SCHEMA {schema}").collect()
-- MAGIC for r in rows:
-- MAGIC     name = r[0]
-- MAGIC     spark.sql(f"DROP POLICY {name} ON SCHEMA {schema}")
-- MAGIC     print("dropped policy:", name)
-- MAGIC print(f"{len(rows)} policies dropped")

-- COMMAND ----------

-- MAGIC %md ## 2. Drop classic row filter / column masks (safe if absent)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # These no-op-error if not set; wrap each so cleanup never fails.
-- MAGIC stmts = [
-- MAGIC     "ALTER TABLE Patients DROP ROW FILTER",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN Email DROP MASK",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN PhoneNumber DROP MASK",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN SSN DROP MASK",
-- MAGIC ]
-- MAGIC for s in stmts:
-- MAGIC     try:
-- MAGIC         spark.sql(s); print("ok:", s)
-- MAGIC     except Exception as e:
-- MAGIC         print("skip:", s, "->", str(e)[:80])

-- COMMAND ----------

-- MAGIC %md ## 3. Remove governance tags

-- COMMAND ----------

-- MAGIC %python
-- MAGIC tag_stmts = [
-- MAGIC     "ALTER TABLE Patients UNSET TAGS ('abac_demo_row')",
-- MAGIC     "ALTER TABLE Visits UNSET TAGS ('abac_demo_row')",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN Email UNSET TAGS ('abac_demo_class')",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN PhoneNumber UNSET TAGS ('abac_demo_class')",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN Address UNSET TAGS ('abac_demo_class')",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN SSN UNSET TAGS ('abac_demo_class')",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN DateOfBirth UNSET TAGS ('abac_demo_class')",
-- MAGIC     "ALTER TABLE Providers ALTER COLUMN Email UNSET TAGS ('abac_demo_class')",
-- MAGIC     "ALTER TABLE Insurance ALTER COLUMN PolicyNumber UNSET TAGS ('abac_demo_class')",
-- MAGIC     "ALTER TABLE Patients ALTER COLUMN Region UNSET TAGS ('abac_demo_geo')",
-- MAGIC     "ALTER TABLE Visits ALTER COLUMN Region UNSET TAGS ('abac_demo_geo')",
-- MAGIC ]
-- MAGIC for s in tag_stmts:
-- MAGIC     try:
-- MAGIC         spark.sql(s); print("ok:", s)
-- MAGIC     except Exception as e:
-- MAGIC         print("skip:", s, "->", str(e)[:80])

-- COMMAND ----------

-- MAGIC %md ## 4. Drop the schema and everything in it
-- MAGIC Comment this cell out if you only want to reset policies but keep the data.

-- COMMAND ----------

DROP SCHEMA IF EXISTS cedip_ws.abac_healthcare_demo CASCADE;

-- COMMAND ----------

-- MAGIC %md ## 5. (Optional) remove account-level persona groups + governed tag keys
-- MAGIC The demo provisioned three account groups and three governed tag keys during setup.
-- MAGIC They are harmless to leave and `00_setup` does NOT recreate them. Drop only when fully
-- MAGIC decommissioning — requires the account-admin profile / account-admin API:
-- MAGIC ```bash
-- MAGIC # account groups
-- MAGIC for g in TEST_GROUP_A TEST_GROUP_B TEST_GROUP_C; do
-- MAGIC   id=$(databricks account groups list -p test-account -o json | \
-- MAGIC        python3 -c "import sys,json;print(next((x['id'] for x in json.load(sys.stdin) if x['displayName']=='$g'),''))")
-- MAGIC   [ -n "$id" ] && databricks account groups delete "$id" -p test-account
-- MAGIC done
-- MAGIC # governed tag keys (workspace host)
-- MAGIC for k in abac_demo_class abac_demo_row abac_demo_geo; do
-- MAGIC   databricks api delete "/api/2.1/tag-policies/$k" -p test-ws
-- MAGIC done
-- MAGIC ```

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Teardown complete. Re-run `00_setup` to rebuild from scratch.
-- MAGIC (Account groups + tag keys persist unless you ran the step above; Carlos in A + C.)

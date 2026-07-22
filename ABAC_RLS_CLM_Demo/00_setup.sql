-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 00 · Setup — Healthcare Governance Demo
-- MAGIC Creates the schema, tables, synthetic data, masking/filter UDFs, and governance tags.
-- MAGIC Run this once. It is **safe to re-run** (idempotent). No policies are created here —
-- MAGIC the data starts fully visible so the RLS/CLM/ABAC reveals in notebooks 01–03 land.
-- MAGIC
-- MAGIC **Persona model:** this workspace's account has three account-level groups —
-- MAGIC `TEST_GROUP_A` (junior/restricted), `TEST_GROUP_B` (senior/full-access),
-- MAGIC `TEST_GROUP_C` (auditor). Carlos is in A + C, not B. Because these are real
-- MAGIC account groups, ABAC policies target them directly with `TO \`TEST_GROUP_A\``
-- MAGIC and the UDFs use `is_account_group_member(...)`.

-- COMMAND ----------

-- MAGIC %md ### One-time persona group provisioning (already done via CLI during setup)
-- MAGIC Created once out-of-band with account-admin credentials against the accounts host.
-- MAGIC NOT re-created on every run. Reproducible record:
-- MAGIC ```bash
-- MAGIC # account-admin profile (accounts.cloud.databricks.com)
-- MAGIC for g in TEST_GROUP_A TEST_GROUP_B TEST_GROUP_C; do
-- MAGIC   databricks account groups create --display-name $g -p test-account
-- MAGIC done
-- MAGIC # add Carlos to A and C (member value = account user id)
-- MAGIC databricks account groups patch <A_id> -p test-account --json \
-- MAGIC   '{"schemas":["urn:ietf:params:scim:api:messages:2.0:PatchOp"],"Operations":[{"op":"add","path":"members","value":[{"value":"<user_id>"}]}]}'
-- MAGIC databricks account groups patch <C_id> -p test-account --json '...same...'
-- MAGIC ```

-- COMMAND ----------

-- MAGIC %md ## 1. Schema

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS cedip_ws.abac_healthcare_demo
  COMMENT 'RLS/CLM/ABAC demo — synthetic healthcare data';
USE CATALOG cedip_ws;
USE SCHEMA abac_healthcare_demo;

-- COMMAND ----------

-- MAGIC %md ## 2. Tables

-- COMMAND ----------

CREATE OR REPLACE TABLE Patients (
    PatientID      STRING NOT NULL,
    FirstName      STRING,
    LastName       STRING,
    DateOfBirth    DATE,
    Gender         STRING,
    PhoneNumber    STRING,
    Email          STRING,
    Address        STRING,
    City           STRING,
    State          STRING,
    ZipCode        STRING,
    Region         STRING,        -- NORTH / SOUTH — drives RLS demo
    SSN            STRING,
    BloodType      STRING,
    CreatedDate    TIMESTAMP
) USING DELTA;

-- COMMAND ----------

CREATE OR REPLACE TABLE Providers (
    ProviderID     STRING NOT NULL,
    FirstName      STRING,
    LastName       STRING,
    Specialty      STRING,
    LicenseNumber  STRING,
    Email          STRING,
    Department     STRING,
    Region         STRING,
    IsActive       BOOLEAN
) USING DELTA;

-- COMMAND ----------

CREATE OR REPLACE TABLE Insurance (
    InsuranceID       STRING NOT NULL,
    PatientID         STRING,
    InsuranceCompany  STRING,
    PolicyNumber      STRING,
    PlanType          STRING,
    Deductible        DECIMAL(10,2),
    IsActive          BOOLEAN
) USING DELTA;

-- COMMAND ----------

CREATE OR REPLACE TABLE Visits (
    VisitID        STRING NOT NULL,
    PatientID      STRING,
    ProviderID     STRING,
    VisitDate      DATE,
    VisitType      STRING,
    Diagnosis      STRING,
    Region         STRING,
    VisitStatus    STRING
) USING DELTA;

-- COMMAND ----------

CREATE OR REPLACE TABLE LabResults (
    LabResultID    STRING NOT NULL,
    VisitID        STRING,
    PatientID      STRING,
    TestName       STRING,
    ResultValue    STRING,
    AbnormalFlag   STRING,
    TestDate       DATE
) USING DELTA;

-- COMMAND ----------

CREATE OR REPLACE TABLE Prescriptions (
    PrescriptionID   STRING NOT NULL,
    VisitID          STRING,
    PatientID        STRING,
    MedicationName   STRING,
    Dosage           STRING,
    PrescriptionDate DATE
) USING DELTA;

-- COMMAND ----------

CREATE OR REPLACE TABLE Billing (
    BillID             STRING NOT NULL,
    PatientID          STRING,
    VisitID            STRING,
    ServiceDescription STRING,
    ChargeAmount       DECIMAL(10,2),
    BalanceDue         DECIMAL(10,2),
    BillingStatus      STRING,
    BillingDate        DATE
) USING DELTA;

-- COMMAND ----------

-- MAGIC %md ## 3. Sample data (synthetic — 12 patients across two regions)

-- COMMAND ----------

INSERT OVERWRITE Patients VALUES
 ('P001','John','Anderson',DATE'1978-03-12','M','415-555-0112','john.anderson@email.com','12 Oak St','San Francisco','CA','94103','NORTH','512-22-1001','O+',current_timestamp()),
 ('P002','Maria','Garcia',DATE'1985-07-24','F','415-555-0198','maria.garcia@email.com','88 Pine Ave','San Francisco','CA','94104','NORTH','512-22-1002','A+',current_timestamp()),
 ('P003','David','Kim',DATE'1990-11-02','M','415-555-0143','david.kim@email.com','5 Elm Rd','Oakland','CA','94607','NORTH','512-22-1003','B-',current_timestamp()),
 ('P004','Susan','Chen',DATE'1972-01-19','F','415-555-0176','susan.chen@email.com','30 Birch Ln','Berkeley','CA','94704','NORTH','512-22-1004','AB+',current_timestamp()),
 ('P005','Robert','Nguyen',DATE'1965-09-30','M','415-555-0121','robert.nguyen@email.com','7 Cedar Ct','San Jose','CA','95112','NORTH','512-22-1005','O-',current_timestamp()),
 ('P006','Linda','Patel',DATE'1988-05-15','F','415-555-0155','linda.patel@email.com','19 Maple Dr','Fremont','CA','94536','NORTH','512-22-1006','A-',current_timestamp()),
 ('P007','James','Wright',DATE'1980-12-08','M','213-555-0110','james.wright@email.com','44 Sunset Blvd','Los Angeles','CA','90028','SOUTH','512-22-1007','B+',current_timestamp()),
 ('P008','Patricia','Lopez',DATE'1993-02-27','F','213-555-0188','patricia.lopez@email.com','61 Palm St','Los Angeles','CA','90015','SOUTH','512-22-1008','O+',current_timestamp()),
 ('P009','Michael','Brown',DATE'1970-06-11','M','619-555-0134','michael.brown@email.com','9 Harbor Dr','San Diego','CA','92101','SOUTH','512-22-1009','A+',current_timestamp()),
 ('P010','Jennifer','Davis',DATE'1982-10-05','F','619-555-0167','jennifer.davis@email.com','23 Ocean Ave','San Diego','CA','92109','SOUTH','512-22-1010','AB-',current_timestamp()),
 ('P011','William','Martinez',DATE'1959-04-22','M','714-555-0143','william.martinez@email.com','77 Grove St','Anaheim','CA','92805','SOUTH','512-22-1011','O+',current_timestamp()),
 ('P012','Elizabeth','Taylor',DATE'1995-08-18','F','714-555-0129','elizabeth.taylor@email.com','2 Valley Rd','Irvine','CA','92602','SOUTH','512-22-1012','B-',current_timestamp());

-- COMMAND ----------

INSERT OVERWRITE Providers VALUES
 ('DR01','Alan','Foster','Cardiology','LIC-88231','alan.foster@hospital.com','Cardiology','NORTH',true),
 ('DR02','Grace','Hill','Pediatrics','LIC-88232','grace.hill@hospital.com','Pediatrics','NORTH',true),
 ('DR03','Omar','Said','Oncology','LIC-88233','omar.said@hospital.com','Oncology','SOUTH',true),
 ('DR04','Nina','Wells','Neurology','LIC-88234','nina.wells@hospital.com','Neurology','SOUTH',true);

-- COMMAND ----------

INSERT OVERWRITE Insurance VALUES
 ('INS001','P001','BlueCross','BC-9928374651','PPO',1500.00,true),
 ('INS002','P002','Aetna','AE-1122938475','HMO',1000.00,true),
 ('INS003','P003','Cigna','CG-5566372819','PPO',2000.00,true),
 ('INS004','P007','UnitedHealth','UH-7788293015','EPO',1250.00,true),
 ('INS005','P009','Kaiser','KP-3344857291','HMO',800.00,true),
 ('INS006','P010','BlueCross','BC-9083746152','PPO',1500.00,true);

-- COMMAND ----------

INSERT OVERWRITE Visits VALUES
 ('V001','P001','DR01',DATE'2025-01-15','Follow-up','Hypertension','NORTH','Completed'),
 ('V002','P002','DR02',DATE'2025-02-03','New','Well child','NORTH','Completed'),
 ('V003','P003','DR01',DATE'2025-02-20','Follow-up','Arrhythmia','NORTH','Completed'),
 ('V004','P007','DR03',DATE'2025-03-01','Consult','Lymphoma screen','SOUTH','Completed'),
 ('V005','P009','DR04',DATE'2025-03-11','New','Migraine','SOUTH','Completed'),
 ('V006','P010','DR03',DATE'2025-03-19','Follow-up','Anemia','SOUTH','Completed'),
 ('V007','P004','DR02',DATE'2025-04-02','Consult','Thyroid','NORTH','Completed'),
 ('V008','P011','DR04',DATE'2025-04-15','Follow-up','Seizure eval','SOUTH','Completed');

-- COMMAND ----------

INSERT OVERWRITE LabResults VALUES
 ('L001','V001','P001','Lipid Panel','LDL 160','H',DATE'2025-01-15'),
 ('L002','V003','P003','ECG','Irregular','H',DATE'2025-02-20'),
 ('L003','V004','P007','CBC','WBC 3.1','L',DATE'2025-03-01'),
 ('L004','V005','P009','MRI Brain','Normal','N',DATE'2025-03-11'),
 ('L005','V006','P010','Ferritin','12 ng/mL','L',DATE'2025-03-19');

-- COMMAND ----------

INSERT OVERWRITE Prescriptions VALUES
 ('RX01','V001','P001','Lisinopril','10mg daily',DATE'2025-01-15'),
 ('RX02','V003','P003','Metoprolol','25mg BID',DATE'2025-02-20'),
 ('RX03','V005','P009','Sumatriptan','50mg PRN',DATE'2025-03-11'),
 ('RX04','V006','P010','Ferrous sulfate','325mg daily',DATE'2025-03-19');

-- COMMAND ----------

INSERT OVERWRITE Billing VALUES
 ('B001','P001','V001','Office visit + labs',450.00,120.00,'Pending',DATE'2025-01-16'),
 ('B002','P002','V002','Well child visit',220.00,0.00,'Paid',DATE'2025-02-04'),
 ('B003','P007','V004','Oncology consult',980.00,450.00,'Pending',DATE'2025-03-02'),
 ('B004','P009','V005','Neurology + MRI',2100.00,600.00,'Pending',DATE'2025-03-12'),
 ('B005','P010','V006','Follow-up + labs',380.00,80.00,'Paid',DATE'2025-03-20');

-- COMMAND ----------

-- MAGIC %md ## 4. Masking & row-filter UDFs
-- MAGIC Deterministic, referential-safe. Used by policies in notebooks 01–03.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION mask_string_partial(s STRING)
  RETURNS STRING
  COMMENT 'Show first char only: John -> J***'
  RETURN CASE WHEN s IS NULL OR length(s) = 0 THEN s
              ELSE concat(substr(s,1,1), '***') END;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION mask_email(e STRING)
  RETURNS STRING
  COMMENT 'Preserve domain only: a@b.com -> ****@b.com'
  RETURN CASE WHEN e IS NULL OR instr(e,'@') = 0 THEN '****'
              ELSE concat('****', substr(e, instr(e,'@'))) END;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION mask_phone(p STRING)
  RETURNS STRING
  COMMENT 'Last 4 digits only: 415-555-0112 -> XXX-XXX-0112'
  RETURN CASE WHEN p IS NULL OR length(p) < 4 THEN 'XXX-XXX-XXXX'
              ELSE concat('XXX-XXX-', substr(p, length(p)-3, 4)) END;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION mask_ssn(s STRING)
  RETURNS STRING
  COMMENT 'Last 4 only: 512-22-1001 -> XXX-XX-1001'
  RETURN CASE WHEN s IS NULL OR length(s) < 4 THEN 'XXX-XX-XXXX'
              ELSE concat('XXX-XX-', substr(s, length(s)-3, 4)) END;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION mask_policy_last4(pn STRING)
  RETURNS STRING
  COMMENT 'Insurance policy last 4: BC-9928374651 -> XXXXXXXX4651'
  RETURN CASE WHEN pn IS NULL OR length(pn) < 4 THEN 'XXXXXXXX'
              ELSE concat('XXXXXXXX', substr(pn, length(pn)-3, 4)) END;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION mask_dob_year(d DATE)
  RETURNS DATE
  COMMENT 'Collapse DOB to Jan 1 of birth year (age band, not exact DOB)'
  RETURN CASE WHEN d IS NULL THEN NULL
              ELSE make_date(year(d), 1, 1) END;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION mask_full(s STRING)
  RETURNS STRING
  COMMENT 'Full redaction: any value -> ***'
  RETURN CASE WHEN s IS NULL THEN NULL ELSE '***' END;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION mask_redact(s STRING)
  RETURNS STRING
  COMMENT 'Redact free-text field -> [REDACTED]'
  RETURN CASE WHEN s IS NULL THEN NULL ELSE '[REDACTED]' END;

-- COMMAND ----------

-- Referential-safe ID mask: same input -> same masked output, so joins still work.
CREATE OR REPLACE FUNCTION mask_id_referential(id STRING)
  RETURNS STRING
  COMMENT 'Deterministic pseudonym preserving joins: P001 -> REF_<crc32>'
  RETURN CASE WHEN id IS NULL THEN NULL
              ELSE concat('REF_', cast(crc32(id) AS STRING)) END;

-- COMMAND ----------

-- Row-filter helper: TRUE only during business hours (9-18 America/Los_Angeles).
CREATE OR REPLACE FUNCTION business_hours_filter()
  RETURNS BOOLEAN
  COMMENT 'TRUE during 9am-6pm Pacific'
  RETURN hour(from_utc_timestamp(current_timestamp(), 'America/Los_Angeles')) BETWEEN 9 AND 17;

-- COMMAND ----------

-- Region row filter: caller sees a region only if they belong to the matching region group.
-- For the demo we map TEST_GROUP_A -> NORTH access; everyone else unrestricted here (Stage 3 tightens).
CREATE OR REPLACE FUNCTION region_filter(region STRING)
  RETURNS BOOLEAN
  COMMENT 'TEST_GROUP_A restricted to NORTH region rows'
  RETURN CASE
           WHEN is_account_group_member('TEST_GROUP_A') THEN region = 'NORTH'
           ELSE TRUE
         END;

-- COMMAND ----------

-- MAGIC %md ## 5. Governance tags (for ABAC stages 02 & 03)
-- MAGIC Tag data once; ABAC `CREATE POLICY` statements in 02/03 then apply masks/filters
-- MAGIC everywhere the tag appears.
-- MAGIC
-- MAGIC **Important:** ABAC `hasTagValue()`/`hasTag()` conditions only accept **governed**
-- MAGIC tag keys (keys backed by an account tag policy). This deployment created three
-- MAGIC dedicated governed keys during setup (see the one-time provisioning block below):
-- MAGIC - `abac_demo_class` (values: pii, phi, redact, dob) — drives column masks
-- MAGIC - `abac_demo_row`   (value: region_scoped) — gates which tables get row-filtered
-- MAGIC - `abac_demo_geo`   (value: region) — marks the column fed to the row-filter function

-- COMMAND ----------

-- MAGIC %md ### One-time governed tag-key provisioning (already done via CLI during setup)
-- MAGIC These keys were created once out-of-band with account-admin credentials. They are NOT
-- MAGIC re-created on every run. Left here (commented) as the reproducible record:
-- MAGIC ```bash
-- MAGIC databricks api post /api/2.1/tag-policies -p test-ws --json '{"tag_key":"abac_demo_class","description":"ABAC demo — data sensitivity class","values":[{"name":"pii"},{"name":"phi"},{"name":"redact"},{"name":"dob"}]}'
-- MAGIC databricks api post /api/2.1/tag-policies -p test-ws --json '{"tag_key":"abac_demo_row","description":"ABAC demo — row scope","values":[{"name":"region_scoped"}]}'
-- MAGIC databricks api post /api/2.1/tag-policies -p test-ws --json '{"tag_key":"abac_demo_geo","description":"ABAC demo — geo column","values":[{"name":"region"}]}'
-- MAGIC ```

-- COMMAND ----------

-- Column masking tags — `abac_demo_class` governs how each sensitive column is masked.
ALTER TABLE Patients ALTER COLUMN Email        SET TAGS ('abac_demo_class' = 'pii');
ALTER TABLE Patients ALTER COLUMN PhoneNumber  SET TAGS ('abac_demo_class' = 'pii');
ALTER TABLE Patients ALTER COLUMN Address      SET TAGS ('abac_demo_class' = 'redact');
ALTER TABLE Providers ALTER COLUMN Email       SET TAGS ('abac_demo_class' = 'pii');

-- COMMAND ----------

-- Highly sensitive STRING columns — full mask.
ALTER TABLE Patients ALTER COLUMN SSN          SET TAGS ('abac_demo_class' = 'phi');
ALTER TABLE Insurance ALTER COLUMN PolicyNumber SET TAGS ('abac_demo_class' = 'phi');
-- DateOfBirth is DATE-typed, so it needs a DATE-returning mask. It gets its own class
-- value (`dob`) because ABAC allows only ONE mask policy per tag value, and all columns
-- sharing a value must be the same type as that value's mask function.
ALTER TABLE Patients ALTER COLUMN DateOfBirth  SET TAGS ('abac_demo_class' = 'dob');

-- COMMAND ----------

-- Row-filter tags:
--   table-level `abac_demo_row=region_scoped` = "apply the region row filter to this table"
--   column-level `abac_demo_geo=region` = "this is the region column to pass to the filter"
ALTER TABLE Patients SET TAGS ('abac_demo_row' = 'region_scoped');
ALTER TABLE Visits   SET TAGS ('abac_demo_row' = 'region_scoped');
ALTER TABLE Patients ALTER COLUMN Region SET TAGS ('abac_demo_geo' = 'region');
ALTER TABLE Visits   ALTER COLUMN Region SET TAGS ('abac_demo_geo' = 'region');

-- COMMAND ----------

-- MAGIC %md ## 6. Baseline check — everything visible, no policies yet

-- COMMAND ----------

SELECT PatientID, FirstName, LastName, Email, PhoneNumber, SSN, Region FROM Patients ORDER BY PatientID;

-- COMMAND ----------

SHOW POLICIES ON SCHEMA cedip_ws.abac_healthcare_demo;

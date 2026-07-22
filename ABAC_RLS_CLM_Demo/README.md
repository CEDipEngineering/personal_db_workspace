# RLS / CLM / ABAC — Unity Catalog Governance Demo

A three-stage, progressively-complex demo of Unity Catalog data governance on synthetic
healthcare data. The pitch: **same query, same table, same instant — different results
based on who you are.**

| Stage | Notebook | Concept |
|-------|----------|---------|
| Setup | `00_setup.sql` | Schema, 7 tables, synthetic data, mask/filter UDFs, governance tags |
| Simple | `01_simple.sql` | **RLS + CLM** — classic per-object row filter + column masks |
| Medium | `02_medium.sql` | **ABAC** — tag data once, one policy governs every matching column/table |
| Complex | `03_complex.sql` | **ABAC orchestration** — multi-role, time-of-day access, join-safe masking |
| Cleanup | `99_cleanup.sql` | Full teardown |

The `.sql` files are in **Databricks notebook source format** — import them as notebooks
(`Workspace → Import → File`) or via the CLI (see below). Cells are delimited by
`-- COMMAND ----------`.

## The narrative

- **01 · Simple.** Attach one row filter (region) and a few column masks (email / phone /
  SSN) to the `Patients` table by hand. Run the same `SELECT` as a restricted vs. full
  user → fewer rows, masked PII. Establishes the primitives.
- **02 · Medium.** Instead of per-object attachment, *tag* columns with a sensitivity
  class and write **one schema-level ABAC policy per class**. The policy applies to every
  tagged column across every table — including tables created later. Tag once, govern
  everywhere. The row filter likewise cascades to any table tagged for row-scoping.
- **03 · Complex.** Overlapping policies for three roles (junior nurse / senior doctor /
  auditor), each seeing different rows, mask depth, and time windows. Includes
  **referential masking** — patient IDs are pseudonymised deterministically, so analytics
  joins still work on masked data — and a `SHOW POLICIES` audit cell.

## Personas

The contrast is driven by three account-level groups:

| Group | Role | Rows | PII | Time |
|-------|------|------|-----|------|
| `TEST_GROUP_A` | Junior nurse | one region only | heavily masked | anytime |
| `TEST_GROUP_C` | Auditor | all regions | last-4 only | business hours only |
| `TEST_GROUP_B` | Senior doctor | all regions | clear | anytime |

Put your demo user in A + C (not B) so you experience the restricted view live, while a
B member is the "sees everything" contrast. Policies target the groups directly
(`TO \`TEST_GROUP_A\``) and UDFs branch on `is_account_group_member(...)`.

## Prerequisites (one-time provisioning)

These are created **once via CLI/API**, not by the notebooks. The notebooks include the
commands as commented records (see the top of `00_setup.sql` and `99_cleanup.sql`).

1. **Account-level persona groups** `TEST_GROUP_A/B/C` (needs account-admin against the
   accounts host — `databricks auth login --host https://accounts.cloud.databricks.com
   --account-id <id> --profile <acct>`). Add your user to A and C.
2. **Governed tag keys** (ABAC `hasTagValue()`/`hasTag()` only accept keys backed by an
   account tag policy):
   - `abac_demo_class` → `pii`, `phi`, `redact`, `dob` (one value per mask type)
   - `abac_demo_row` → `region_scoped` (table-level row-filter gate)
   - `abac_demo_geo` → `region` (marks the column fed to the row filter)

## Run it

```bash
# import the notebooks
for f in 00_setup 01_simple 02_medium 03_complex 99_cleanup; do
  databricks workspace import /Users/<you>/abac_demo/$f \
    --file $f.sql --language SQL --format SOURCE --overwrite -p <profile>
done
```

Then run `00_setup` once, and walk `01 → 02 → 03` live (each notebook resets prior-stage
policies in its first `%python` cell, so they're independently re-runnable). `99_cleanup`
tears everything down.

`run.py` is a local validation helper that executes a stage's SQL cells via the SQL
Statements API (skips `%python`/`%md`) — handy for a quick sanity check before importing.
Set the constants at the top for your workspace.

## Portability notes (learned deploying this across workspaces)

- **ABAC needs governed tag keys.** Ungoverned/custom keys are rejected with
  "Unknown tag policy key". If you can't create keys (account tag-policy limit), reuse
  existing governed keys with enough distinct values.
- **One mask policy per tag value, and the column type must match the mask function's
  argument type.** Two masks on one column → `MULTIPLE_MASKS` error; a type-mismatched
  mask is *silently skipped* and leaks real data. Keep DATE columns (e.g. DateOfBirth) on
  their own tag value with a DATE-returning mask.
- **Row-filter policy syntax:** `ROW FILTER fn TO <principal> FOR TABLES WHEN
  hasTagValue(<tableTag>) MATCH COLUMNS hasTag(<colTag>) AS r USING COLUMNS(r)`. The
  filtered column must itself be tagged; you can't name a raw column in `USING COLUMNS`.
- **`DROP POLICY` has no `IF EXISTS`.** Each stage opens with a `%python` cell that
  enumerates `SHOW POLICIES` and drops each — makes stages re-runnable.
- **Admin/table-owner does NOT bypass RLS/CLM** — the restricted view applies to you even
  as workspace admin + table owner, so the live self-contrast works.
- **Non-federated workspaces:** if the workspace isn't identity-federated, workspace SCIM
  groups can't be ABAC principals and `is_account_group_member()` won't see them. Use
  account-level groups (or, as a fallback, target `account users` and branch inside the
  UDF with `is_member('<workspace_group>')`).

## Data

Fully synthetic. 12 patients across two regions (NORTH/SOUTH) plus providers, insurance,
visits, lab results, prescriptions, and billing. No real PII.

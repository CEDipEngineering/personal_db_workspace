#!/usr/bin/env python3
"""Dry-run a notebook-source .sql file cell-by-cell via the Databricks SQL Statements API.
Splits on `-- COMMAND ----------`, strips the notebook header, runs each non-empty cell.
Skips %python / %md cells (import those into a notebook and run them there).
Usage: python3 run.py <file.sql> [--catalog C] [--schema S]

This is a local dev/validation helper — the notebooks themselves are the deliverable.
Set the four constants below (or pass --catalog/--schema) for your workspace, then run
each stage in order to sanity-check before importing to Databricks.
"""
import json, subprocess, sys, time

PROFILE = "test-ws"                 # databricks CLI profile
WAREHOUSE = "64d4d2b540c40422"      # a running SQL warehouse id
CATALOG = "cedip_ws"               # target catalog
SCHEMA = "abac_healthcare_demo"    # created by 00_setup

def run_sql(stmt, catalog, schema):
    payload = {
        "warehouse_id": WAREHOUSE,
        "catalog": catalog,
        "schema": schema,
        "statement": stmt,
        "wait_timeout": "50s",
    }
    p = subprocess.run(
        ["databricks", "api", "post", "/api/2.0/sql/statements", "-p", PROFILE,
         "--json", json.dumps(payload)],
        capture_output=True, text=True)
    if p.returncode != 0:
        return {"status": {"state": "CLIENT_ERROR"}, "raw": p.stderr}
    try:
        r = json.loads(p.stdout)
    except Exception:
        return {"status": {"state": "PARSE_ERROR"}, "raw": p.stdout}
    # poll if still running
    sid = r.get("statement_id")
    while r.get("status", {}).get("state") in ("PENDING", "RUNNING") and sid:
        time.sleep(2)
        pp = subprocess.run(
            ["databricks", "api", "get", f"/api/2.0/sql/statements/{sid}", "-p", PROFILE],
            capture_output=True, text=True)
        r = json.loads(pp.stdout)
    return r

def main():
    path = sys.argv[1]
    catalog, schema = CATALOG, SCHEMA
    if "--catalog" in sys.argv: catalog = sys.argv[sys.argv.index("--catalog")+1]
    if "--schema" in sys.argv: schema = sys.argv[sys.argv.index("--schema")+1]
    text = open(path).read()
    cells = text.split("-- COMMAND ----------")
    n_ok = n_fail = 0
    for i, cell in enumerate(cells):
        # strip notebook header and comment-only lines
        lines = [l for l in cell.splitlines()
                 if l.strip() and not l.strip().startswith("-- Databricks notebook source")
                 and not l.strip().startswith("-- MAGIC")]
        body = "\n".join(lines).strip()
        # remove pure-comment cells
        noncomment = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("--")]
        if not noncomment:
            continue
        # a cell may hold multiple statements separated by ';' — split on ';\n'
        stmts = [s.strip() for s in body.split(";\n") if s.strip() and any(
            not ln.strip().startswith("--") for ln in s.splitlines())]
        # fallback: if no ';\n', treat whole cell as one
        if not stmts:
            stmts = [body]
        for stmt in stmts:
            stmt = stmt.rstrip(";").strip()
            if not stmt or all(l.strip().startswith("--") for l in stmt.splitlines()):
                continue
            r = run_sql(stmt, catalog, schema)
            state = r.get("status", {}).get("state")
            label = stmt.splitlines()[0][:70]
            if state == "SUCCEEDED":
                n_ok += 1
                # show small result preview
                data = r.get("result", {}).get("data_array")
                if data and len(data) <= 12:
                    print(f"[cell {i}] OK: {label}\n     -> {data}")
                else:
                    rows = len(data) if data else 0
                    print(f"[cell {i}] OK: {label}  ({rows} rows)")
            else:
                n_fail += 1
                err = r.get("status", {}).get("error", {}).get("message") or r.get("raw", "")
                print(f"[cell {i}] FAIL ({state}): {label}\n     !! {err[:400]}")
    print(f"\n=== {path}: {n_ok} ok, {n_fail} failed ===")

if __name__ == "__main__":
    main()

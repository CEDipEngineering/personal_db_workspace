# Databricks notebook source
# MAGIC %md
# MAGIC # PDF OCR Extraction and AI-Powered Data Structuring in Databricks
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC 1. OCR-based PDF extraction (using **Docling**)
# MAGIC 2. Table structure reconstruction
# MAGIC 3. AI-based summarization of PDF contents
# MAGIC 4. AI-powered schema transformation of table data into structured Spark DataFrames
# MAGIC 5. Persisting structured results for querying and analytics
# MAGIC
# MAGIC **References**:
# MAGIC - https://docling-project.github.io/docling/examples/minimal/
# MAGIC - https://docs.databricks.com/aws/en/machine-learning/model-serving/structured-outputs
# MAGIC - https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required dependencies

# COMMAND ----------

# MAGIC %pip install pandas docling
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports and initialization

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. PDF Parsing Function

# COMMAND ----------

def main(input_doc_path_list: str):
    # Define processing pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Create the converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    settings.debug.profile_pipeline_timings = True

    # Convert the documents
    conversion_result = converter.convert_all(input_doc_path_list)
    output = []

    for res in conversion_result:
        doc = res.document
        raw_document_markdown = doc.export_to_markdown()

        tables_markdown = "\n".join([
            f"Table {table_ix}:\n{table.export_to_dataframe().to_markdown()}"
            for table_ix, table in enumerate(doc.tables)
        ])

        doc_conversion_secs = res.timings["pipeline_total"].times

        output.append({
            "document": doc.name,
            "num_pages": doc.num_pages(),
            "full_markdown": raw_document_markdown,
            "tables_markdown": tables_markdown,
            "total_conversion_time_secs": doc_conversion_secs[0],
        })

    return output

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load and Process Local PDF Files

# COMMAND ----------

# Using local files from Workspace directory — recommend using Volumes for persistence
data_folder = Path("./assets").resolve()
input_doc_paths = list(data_folder.glob("*"))

results = main([str(input_doc_path) for input_doc_path in input_doc_paths])
display(pd.DataFrame(results))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save Extracted Data to a Databricks Table

# COMMAND ----------

df = spark.createDataFrame(pd.DataFrame(results))
df.write.mode("overwrite").saveAsTable("main.default.pdf_conversion_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. AI-based Summarization & Structured Table Output

# COMMAND ----------

df = spark.sql("""
SELECT
    document,
    full_markdown,
    ai_query(
        'databricks-gpt-oss-120b',
        'Summarize the following document content:\n\n' || full_markdown
    ) AS ai_summary,
    ai_query(
        'databricks-claude-3-7-sonnet',
        'Restructure and organize the content of the following tables to fit the provided schema, you must analyze the tables, and summarize their expenditures and profits.' ||
        '''
        Please adhere to this schema:
        schema = StructType([
            StructField("table_summary", ArrayType(
                StructType([
                    StructField("table_index", IntegerType(), True),
                    StructField("expenditures", DoubleType(), True),
                    StructField("profits", DoubleType(), True),
                    StructField("notes", StringType(), True),
                ])
            ), True),
            StructField("total_expenditures", DoubleType(), True),
            StructField("total_profits", DoubleType(), True),
        ])
        ''' || tables_markdown,
        responseFormat => '
        STRUCT<
          result: STRUCT<
            total_expenditures:DOUBLE,
            total_profits:DOUBLE,
            table_summary:ARRAY<STRUCT<table_index:INT, expenditures:DOUBLE, profits:DOUBLE, notes:STRING>>
          >
        >
        '
    ) AS asset_financials
FROM main.default.pdf_conversion_results
""")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Parse AI JSON Output into Structured DataFrame

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, IntegerType, StringType

# Define the nested schema matching your JSON output
schema = StructType([
    StructField("table_summary", ArrayType(
        StructType([
            StructField("table_index", IntegerType(), True),
            StructField("expenditures", DoubleType(), True),
            StructField("profits", DoubleType(), True),
            StructField("notes", StringType(), True),
        ])
    ), True),
    StructField("total_expenditures", DoubleType(), True),
    StructField("total_profits", DoubleType(), True),
])

# Assuming your JSON string column is `asset_financials` (type: string), parse it with from_json:
df_structured = df.withColumn(
    "asset_financials_struct",
    F.from_json(F.col("asset_financials"), schema)
)
display(df_structured)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Flatten Table Summaries for Analysis

# COMMAND ----------

df_structured.select(
    "document",
    F.explode(F.col("asset_financials_struct")["table_summary"]).alias("col")
).withColumns({
    "expenditures": F.col("col.expenditures"),
    "profits": F.col("col.profits"),
    "notes": F.col("col.notes"),
    "table_index": F.col("col.table_index"),
}).drop("col").display()


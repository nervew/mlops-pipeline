# Databricks notebook source
# MAGIC %md
# MAGIC Revisión existencia MLFlow

# COMMAND ----------

import mlflow
print(mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC Revisión existencia librerias utiles

# COMMAND ----------

# MAGIC %pip install scikit-learn pandas

# COMMAND ----------

# %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC Creación de espacio de trabjao MLFlow - Como pruebas

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Shared/mlops-pipeline")

# COMMAND ----------

# MAGIC %md
# MAGIC Verificamos la integración

# COMMAND ----------

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)

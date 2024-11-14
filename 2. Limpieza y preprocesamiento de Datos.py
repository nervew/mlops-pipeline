# Databricks notebook source
# MAGIC %md
# MAGIC # Limpieza y preprocesamiento de Datos

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variables

# COMMAND ----------

# Definir umbral de tolerancia (por ejemplo, eliminar columnas con > 50% valores nulos)
porcentaje_tolerancia_nulos_columna = 0.5

# Definir el percentil superior y el percentil inferior para eliminar como outiers para las columnas
percentil_inferior = 0.05
percentil_superior = 0.95

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importar liberías

# COMMAND ----------

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lectura de los datos

# COMMAND ----------

df = spark.sql("SELECT * FROM hive_metastore.db_mipres_suministro.t_municipio").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manejo de valores nulos

# COMMAND ----------

# DBTITLE 1,Identificar Valores Nulos
# Contar valores nulos por columna
null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0]
null_counts

# COMMAND ----------

# DBTITLE 1,Visualizar Valores Nulos
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Mapa de Valores Nulos")
plt.show()

# COMMAND ----------

# DBTITLE 1,Eliminar Columnas con Demasiados Valores Nulos
threshold = len(df) * porcentaje_tolerancia_nulos_columna
df = df.dropna(thresh=threshold, axis=1)

# COMMAND ----------

# DBTITLE 1,Imputar Valores Nulos
# Imputar con la media para variables numéricas
from sklearn.impute import SimpleImputer

numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer_num = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

# Imputar con la moda para variables categóricas
categorical_cols = df.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# COMMAND ----------

# DBTITLE 1,Verificación de No existencia de Valores Nulos
total_nulls = df.isnull().sum().sum()
print(f"Total de valores nulos después de imputación: {total_nulls}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eliminación de Filas Duplicadas

# COMMAND ----------

# DBTITLE 1,Contar y Eliminar Duplicados
# Contar filas duplicadas
duplicated_rows = df.duplicated().sum()
print(f"Filas duplicadas: {duplicated_rows}")

# Eliminar filas duplicadas
df = df.drop_duplicates()

# COMMAND ----------

# DBTITLE 1,Verificar Eliminación
duplicated_rows_post = df.duplicated().sum()
print(f"Filas duplicadas después de eliminación: {duplicated_rows_post}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manejo de Outliers

# COMMAND ----------

# DBTITLE 1,Identificar Outliers Utilizando Boxplot
# Boxplot
for var in numeric_cols:
  plt.figure(figsize=(8,6))
  sns.boxplot(x=df[var])
  plt.title("Boxplot de "+var)
  plt.show()

# COMMAND ----------

# DBTITLE 1,Eliminar Outliers
# Definir límites
for var in numeric_cols:
  Q1 = df[var].quantile(percentil_inferior)
  Q3 = df[var].quantile(percentil_superior)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Filtrar outliers
  df = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)]

# COMMAND ----------

# DBTITLE 1,Verificar Eliminación de Outliers
# Boxplot actualizado
for var in numeric_cols:
  plt.figure(figsize=(8,6))
  sns.boxplot(x=df[var])
  plt.title("Boxplot de "+var+" Después de Eliminación de Outliers")
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalización y Escalado

# COMMAND ----------

# DBTITLE 1,Seleccionar Variables a Escalar
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Variables numéricas a escalar: {numeric_cols}")

# COMMAND ----------

def seleccionar_escalador(df, umbral_asimetria=0.5):
    """
    Selecciona automáticamente el método de escalado basado en la asimetría de los datos.

    Parámetros:
    - df: DataFrame con las columnas numéricas a escalar.
    - umbral_asimetria: Valor absoluto de asimetría para decidir el tipo de escalado.

    Retorna:
    - Un diccionario con el escalador seleccionado para cada columna.
    """
    escaladores = {}
    for columna in df.columns:
        skewness = df[columna].skew()
        if abs(skewness) < umbral_asimetria:
            escaladores[columna] = StandardScaler()
        else:
            escaladores[columna] = MinMaxScaler()
    return escaladores


# COMMAND ----------

# DBTITLE 1,Amplir Escalado (Standar Scaling)
# Obtener los escaladores para cada columna
escaladores_seleccionados = seleccionar_escalador(df[numeric_cols])

# Crear una copia del DataFrame para escalado
df_escalado = df.copy()

# Aplicar el escalado correspondiente a cada columna
for columna in numeric_cols:
    scaler = escaladores_seleccionados[columna]
    df_escalado[[columna]] = scaler.fit_transform(df[[columna]])

# Reemplazar el DataFrame original con el escalado
df[numeric_cols] = df_escalado[numeric_cols]


# COMMAND ----------

# DBTITLE 1,Verificar el Escalado
df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardar el DataFrame Preprocesado

# COMMAND ----------

# MAGIC %md
# MAGIC ## Revisiones PostProcesamiento

# COMMAND ----------

# DBTITLE 1,Visualizar Distribución de Variables Escaladas
df.hist(bins=50, figsize=(20,15))
plt.suptitle("Distribuciones Después del Escalado")
plt.show()

# COMMAND ----------

# DBTITLE 1,Mapa de Calor de Correlaciones Actualizado
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de Correlaciones Después del Preprocesamiento")
plt.show()

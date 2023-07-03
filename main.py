#Passo 1: Preparação do ambiente
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
import pandas as pd

#Passo 2: Configuração do Spark
spark = SparkSession.builder.appName("DemandPrediction").getOrCreate()

#Passo 3: Carregamento dos dados
data = spark.read.csv("caminho/do/arquivo.csv", header=True, inferSchema=True)

#Passo 4: Pré-processamento dos dados
data = data.withColumn("timestamp", data["data"].cast("timestamp"))

#Passo 5: Engenharia de recursos
data = data.withColumn("mes", data["timestamp"].substr(6, 2).cast("integer"))

#Passo 6: Divisão dos dados
train_data, test_data = data.randomSplit([0.8, 0.2])

#Passo 7: Construção do modelo RandomForestRegressor do PySpark:
assembler = VectorAssembler(inputCols=["mes"], outputCol="features")
train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

rf = RandomForestRegressor(featuresCol="features", labelCol="demanda")
model = rf.fit(train_data)

#Passo 8: Avaliação do modelo
predictions = model.transform(test_data)
predictions.select("mes", "demanda", "prediction").show()

#Passo 9: Visualização dos resultados
df_predictions = predictions.select("mes", "demanda", "prediction").toPandas()
df_predictions.plot(x="mes", y=["demanda", "prediction"], kind="line")


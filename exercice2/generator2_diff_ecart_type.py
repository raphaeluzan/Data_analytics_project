# =============================================
# On genere les donnees
# =============================================

# imports
import sys
import random
import numpy
from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs

# var
val_min = 0
val_max = 100
ETAPES = 0.1


def point_valeurs(valeur_moyenne, valeur_normale, dev, cluster, dim):
    valeurs = ""
    for d in range(dim):
        valeur = valeur_moyenne[d] + valeur_normale[d] * dev
        if not valeurs:
            valeurs = str(valeur)
        else:
            valeurs = valeurs + "," + str(valeur)
    return (valeurs + "," + str(cluster))



pts = int(30) # number of pts to be generated
k = int(3) # number of rdd
dim = int(4) # dim of the data


for i in range(1,100):
    dev = int(i)
    file_name = "out_dev_"+ str(i) + '.csv'
    rdd = sc.parallelize(range(0, k))
    clust_mean = rdd.map(lambda cluster : (cluster, random.sample(list(numpy.arange(val_min, val_max, ETAPES)), dim)))
    valeurs_vector_alea = RandomRDDs.normalVectorRDD(sc, numRows = pts, numCols = dim, numPartitions = k, seed = 1)
    # assiging a random cluster for each point
    cluster_valeur_normales_vector = valeurs_vector_alea.map(lambda point : (random.randint(0, k - 1), point.tolist()))
    # generate a valeur depending of the mean of the cluster, standard deviation and the normal valeur 
    pts_valeur_vector = cluster_valeur_normales_vector.join(clust_mean).map(lambda x: (point_valeurs(x[1][1], x[1][0], dev, x[0], dim)))
    #Voir le resultat
    print(pts_valeur_vector.collect())
    # writing pts valeur in a 1 csv file
    # write_into_csv(file_name, pts_valeur_vector);
    # saving rdd using saveAsTextFile  
    pts_valeur_vector.saveAsTextFile(file_name)


# =============================================
# On calcul le wsse
# =============================================
#Imports
from pyspark.mllib.random import RandomRDDs
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassparklassificationEvaluator
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
import numpy as np
import datetime
import sys
from pyspark.ml.clustering import KMeans
#Imports
from pyspark.mllib.random import RandomRDDs
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType
from pyspark import SparkContext
from random import randint
import numpy as np
import datetime
import sys
from statistics import mean
from statistics import stdev
from statistics import mode
from statistics import median
import random

# Create spark session
#spark = SparkSession.builder.master("local").appName("iris_classification").getOrCreate()
#sc = SparkContext("local", "generator") 
spark = SparkSession.builder.master("local").appName("iris").getOrCreate()


#Variables
path1 = "out.csv/part-0000*"
path2 = "out2.csv/part-0000*"


colnames = ["x1", "x2", "x3", "x4"]

a = [ StructField(colname, FloatType(), False) for colname in colnames ] 
a.append(StructField("label", StringType(), False))

schema = StructType ( a )

# assembler group all x1..x2 into a single col called X
assembler = VectorAssembler( inputCols = colnames, outputCol="features" )



# TRAINING 
res = []
for i in range(1,100):
    # load the data into the dataframe
    path = "out_dev_"+ str(i) + '.csv/part-*'
    df = spark.read.csv(path, schema = schema)
    df.show(truncate=False)
    df = assembler.transform(df) #group all x1..x2 into a single col called X
    # keep X and y only
    df = df.select("features", "label")
    print("sparkhema: ")
    #df.printsparkhema()
    print("Data")
    #df.show(truncate=False)
    kmeans = KMeans().setK(3).setSeed(randint(0,2000))
    model = kmeans.fit(df)
    wssse = model.computeCost(df)
    print("Within Set Sum of Squared Errors = " + str(wssse))
    res.append(wssse)
    print(res)

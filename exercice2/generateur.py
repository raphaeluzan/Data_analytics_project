# spark-submit generator.py out 9 3 2 10
# spark-submit kmeans.py 'out.csv/part-*' 9 10
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

if len(sys.argv) != 6:
    print("Essayez : spark-submit generator.py out 9 3 2 10")
    exit(0)


# inputs
file_name = sys.argv[1] + '.csv'  # Nom du fichier
pts = int(sys.argv[2]) # Nbr de pts generes
k = int(sys.argv[3]) # nombre k de cluster
dim = int(sys.argv[4]) # dim
dev = int(sys.argv[5]) # ecart-type souhaite

# méthodes
def point_valeurs(valeur_moyenne, valeur_normale, dev, cluster, dim):
    valeurs = ""
    for d in range(dim):
        valeur = valeur_moyenne[d] + valeur_normale[d] * dev
        if not valeurs:
            valeurs = str(valeur)
        else:
            valeurs = valeurs + "," + str(valeur)
    return (valeurs + "," + str(cluster))


#code
'''
# inputs
file_name = "out" + '.csv'  # file name to be generated
pts = int(9) # number of pts to be generated
k = int(3) # number of rdd
dim = int(2) # dim of the data
dev = int(10) # standard deviation
'''

sc = SparkContext("local", "generator") # spark context
rdd = sc.parallelize(range(0, k))
clust_mean = rdd.map(lambda cluster : (cluster, random.sample(list(numpy.arange(val_min, val_max, ETAPES)), dim)))
valeurs_vector_alea = RandomRDDs.normalVectorRDD(sc, numRows = pts, numCols = dim, numPartitions = k, seed = 1)
cluster_valeur_normales_vector = valeurs_vector_alea.map(lambda point : (random.randint(0, k - 1), point.tolist()))
pts_valeur_vector = cluster_valeur_normales_vector.join(clust_mean).map(lambda x: (point_valeurs(x[1][1], x[1][0], dev, x[0], dim)))

#Voir le resultat
print(pts_valeur_vector.collect())

# on enregistre le rdd
pts_valeur_vector.saveAsTextFile(file_name)

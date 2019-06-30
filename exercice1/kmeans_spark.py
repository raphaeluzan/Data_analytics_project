# commande : spark-submit kmeans.py data/data.csv k m
# commande exemple : spark-submit kmeans.py data/data.csv 3 10
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


# Create spark session
#spark = SparkSession.builder.master("local").appName("iris_classification").getOrCreate()
sc = SparkContext("local", "generator") 
'''
#Variables
path = "/Users/raphaeluzan/spark-kmeans-kmeansPP/data/iris_clustering.dat"
nbrClusters = 3
maxIterations = 10
commande : spark-submit kmeans.py data.csv k m
'''


if len(sys.argv) != 4:
	print("Vous devez entrer 3 arguments")
	print(" 1 fichier contenant les donnees")
	print(" 2 Nombre k de cluster")
	print(" 3 Nombre d'iterations \n")
	print("commande : spark-submit kmeans.py data/iris_small.dat 4 10")
	exit(0)

# inputs
path = sys.argv[1]  # file name of the points
nbrClusters = int(sys.argv[2]) # number of clusters
maxIterations = int(sys.argv[3]) # maximum number of iterations


#Fonctions
def loadData(path):
	data = sc.textFile(path)
	datawithoutEmpty = data.filter(lambda x: x != "").filter(lambda x: x is not None)
	datawithIndex = datawithoutEmpty.map(lambda x : x.split(',')).zipWithIndex()
	return datawithIndex.map(lambda x : (x[1],x[0]))

def initCentroids_withsample(data, nbrClusters):
    sample = sc.parallelize(data.takeSample(False, nbrClusters,seed=randint(0, 2000)))
    centroids = sample.map(lambda point : point[1][:-1])
    return centroids.zipWithIndex().map(lambda point : (point[1], point[0]))

'''
def initCentroids_with_random_rdd(data):
	rdd = rdd.map(lambda x : (float(x[1][0]),float(x[1][1]),float(x[1][2]),float(x[1][3])))
	colnames = ["x0", "x1", "x2", "x3" ]
	a = [ StructField(colname, FloatType(), False) for colname in colnames ]
	schema = StructType (a)
	df_to_avg = spark.createDataFrame(rdd, schema)
	res[x0_max]= df_to_avg.agg({"x0": "max"}).collect()[0]["max(x0)"] #get max of x0
    mean = 2.0
	x = RandomRDDs.normalVectorRDD(sc, 3,4, seed=1)
	return x.collect()
'''
def distance_minimum(elem):
	myList = elem[1]
	minValue = -1
	point_min = None
	minCentroidIndex = -1
	for element in myList:
		if (minValue == -1) or (element[1] < minValue):
			minValue = element[1]
			minCentroidIndex = element[0]
			point_min = (minCentroidIndex, minValue)
	return (elem[0], point_min)

def assignToCluster(rdd, centroids): 
	cartesian = centroids.cartesian(rdd)
	rdd_all_distance = cartesian.map(lambda x: (x[1][0], (x[0][0],distance(x[0][1],x[1][1][:-1]))))
	return rdd_all_distance.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda elem: distance_minimum(elem))

def distance(x,y):
	return np.linalg.norm(np.array(x, dtype='f')-np.array(y, dtype='f'))

def calculate_barycentre(num_centroid, rdd_item):
	res = []
	for element in rdd_item:
		res.append(element[0][:-1])
	centroids_new =(num_centroid,list(np.average(np.array(res).astype(np.float), axis = 0)))
	return centroids_new

def computeCentroids(rdd_item):
	rdd_item_with_join = rdd_item.join(rdd).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))).groupByKey()
	rdd_item_with_join = rdd_item_with_join.map(lambda x: (x[0], list(x[1])))
	new_centroids = rdd_item_with_join.map(lambda x: calculate_barycentre(x[0], x[1]))
	return new_centroids

def computeIntraClusterDistance(rdd_item):
	rdd_points = rdd_item.map(lambda x : x[1])
	count_elem_by_cluster = rdd_points.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b : a + b)
	sum_elem_by_cluster = rdd_points.reduceByKey(lambda x, y: x + y)
	distance_moyenne = sum_elem_by_cluster.join(count_elem_by_cluster).map(lambda x: x[1][0]/x[1][1])
	res = distance_moyenne.sum()
	return res

def converge(centroids, rdd_new_centroids):
	res = centroids.join(rdd_new_centroids)
	bool_converge = res.map(lambda x: np.array_equal(x[1][0], x[1][1]))
	bool_res = bool_converge.filter(lambda x: x == False).count() > 0
	return not bool_res
# code

rdd = loadData(path)



rdd_centroids = initCentroids_withsample(rdd, nbrClusters)
seuil_conv=0.01
i = 0

historique_distance = []
startTime = datetime.datetime.now()
while i < maxIterations:
	rdd_assignToCluster = assignToCluster(rdd, rdd_centroids) # On affecte a chaque elem son centroide le plus proche
	new_centroids = computeCentroids(rdd_assignToCluster) # on recalcul le centre de chacun de nos cluster
	intraClusterDistances = computeIntraClusterDistance(rdd_assignToCluster) #calcule la distance intra cluster d’une affectation
	historique_distance.append(intraClusterDistances)
	i += 1
	print("******** =>. L'iteration numero  #" + str(i) + ' donne la distance : ' + str(intraClusterDistances))
	if i >1:
		if converge(rdd_centroids, new_centroids):
			print("FIN ******** =>. L'iteration numero  #" + str(i) + ' donne la distance : ' + str(intraClusterDistances))
			break;
	rdd_centroids = sc.parallelize(new_centroids.collect())

endTime = datetime.datetime.now()
print("**********************************")
print("Temps d'execution total: " + str(endTime - startTime))
print("Nomrbe d'iteration pour converger: " + str(i))
print("Final distance: " + str(intraClusterDistances))
print("**********************************")

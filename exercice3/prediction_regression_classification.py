# imports
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
# Create spark session
spark = SparkSession.builder.master("local").appName("Student").getOrCreate()


#load data
rdd = sc.textFile("/Users/raphaeluzan/Downloads/FormatStudentPerformanceClassification.csv")
rdd = rdd.map(lambda x:  [ float(i) for i in x.split(',')] )
test_valeur = 6
colnames = []
for i in range(7) :
	if i==test_valeur:
		colnames.append(str("label"))
	else:
		colnames.append(str("x"+str(i)))

schema = StructType ([ StructField(colname, FloatType(), False) for colname in colnames])
df = spark.createDataFrame(rdd, schema)
#df.show()
assembler = VectorAssembler( inputCols = colnames[:-1], outputCol="features" )
df = assembler.transform(df)
df = df.select("features", "label")
train, test = df.randomSplit([0.6, 0.4], 22)

train.show(truncate=False)
comparaison = {}

print("###############")
print("*** Kmeans ***")
print("###############")
classifier = KMeans().setK(2).setSeed(1)
model=classifier.fit(train)
predictions = model.transform(test)
predictions.groupBy("prediction","label").count().show()
predictions2 = predictions.withColumn("prediction2", predictions["prediction"].cast(DoubleType()))

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction2", metricName="accuracy")
accuracy = evaluator.evaluate(predictions2)
print("Test Error = %g " % (1.0 - accuracy))
comparaison["kmeans"] = accuracy

print("###############")
print("*** RegressionLogistique ***")
print("###############")
classifier= LogisticRegression()
model=classifier.fit(train)
predictions = model.transform(test)
predictions.groupBy("prediction","label").count().show()
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))
comparaison["regression_logistique"] = accuracy


print("###############")
print("*** RegressionLogistique Avec seuil optimise***")
print("###############")
from pyspark.ml.classification import LogisticRegression
# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = model.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# Set the model threshold to maximize F-Measure
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
    .select('threshold').head()['threshold']
classifier.setThreshold(bestThreshold)

model=classifier.fit(train)
predictions = model.transform(test)
predictions.groupBy("prediction","label").count().show()
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))
comparaison[regression_logistique_param] = accuracy


print(comparaison)


print("###############")
print("*** DecisionTreeClassifier ***")
print("###############")
# Split the data into training and test sets (30% held out for testing)
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)
# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
# Train model.  This also runs the indexers.
model = pipeline.fit(train)
# Make predictions.
predictions = model.transform(test)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))
comparaison["DecisionTreeClassifier"] = accuracy
predictions.groupBy("prediction","indexedLabel").count().show()


treeModel = model.stages[1]
treeModel = model.stages[2]
# summary only
#print(treeModel)

print("###############")
print("*** SVC ***")
print("###############")
from pyspark.ml.classification import LinearSVC


lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the model
model = lsvc.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))






#################


# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

## Exemple d'évaluator
'''
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
'''
##### PB de type regression
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rdd = sc.textFile("/Users/raphaeluzan/Downloads/FormatStudentPerformanceRegression.csv")
rdd = rdd.map(lambda x:  [ float(i) for i in x.strip().split(',')] )


test_valeur = 8
colnames = []
for i in range(9) :
	if i==test_valeur:
		colnames.append(str("label"))
	else:
		colnames.append(str("x"+str(i)))

schema = StructType ([ StructField(colname, FloatType(), False) for colname in colnames])
df = spark.createDataFrame(rdd, schema)

comparaison = {}


assembler = VectorAssembler( inputCols = colnames[:-1], outputCol="features" )
df = assembler.transform(df)
df.show()
df = df.select("features", "label")
train, test = df.randomSplit([0.7, 0.3],seed=1)
train.show()


# instantiate the base classifier.
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

# instantiate the One Vs Rest Classifier.
ovr = OneVsRest(classifier=lr)

# train the multiclass model.
ovrModel = ovr.fit(train)

# score the model on test data.
predictions = ovrModel.transform(test)

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# compute the classification error on test data.
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))


from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
comparaison["OneVsRest"] = rmse


### regression 
### regression lineaire
### regression lineaire
### regression lineaire
### regression lineaire
from pyspark.ml.regression import LinearRegression

# Load training data

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(train)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

comparaison["LinearRegression"] = trainingSummary.rootMeanSquaredError
###
from pyspark.ml.classification import NaiveBayes
# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


comparaison["NaiveBayes"] = rmse

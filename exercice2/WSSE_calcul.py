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




# load the data into the dataframe
path = "grand" + '.csv/part-*'
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


wssse1 = "39.5368979345"
wssse2 = "103.9467530928"



print("WSSE avec deux centroide d'écart-type 1")
print("Within Set Sum of Squared Errors = " + str(wssse1))


print("WSSE avec un centroide d'écart-type 1 et l'autre d'écart-type 4")
print("Within Set Sum of Squared Errors = " + str(wssse2))




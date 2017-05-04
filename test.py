from pyspark import SparkContext
from clustering import KHMean

sc = SparkContext(appName="nyc_taxi_data")

data_path = ''  # Enter the path of the data file here

features_rdd = sc.textFile(data_path)  # Create the RDD. All the features must of floating point numbers or integers

k = 30  # No. of clusters
n = 100  # No. of iterations
model = KHMean()  # Call the model
centers = model.train(features_rdd, k, n)  # Train the model

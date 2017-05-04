from pyspark import SparkContext
from clustering import KHMean

sc = SparkContext(appName="nyc_taxi_data")

data_path = ''  # Enter the path of the data file here

data_rdd = sc.textFile(data_path)

k = 30
model = KHMean()
centers = model.train(data_rdd, k, 100)

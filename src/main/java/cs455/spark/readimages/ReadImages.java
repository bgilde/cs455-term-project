package cs455.spark.readimages;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.*;
import org.apache.spark.ml.*;
import org.apache.spark.ml.image.*;
import scala.Tuple2;

import java.util.Arrays;
import java.util.Iterator;


public class ReadImages {
  public static void main(String[] args) {

   // Define a configuration to use to interact with Spark
   SparkConf conf = new SparkConf().setAppName("Image saver");

   // read images from hdfs
   Dataset<Row> ds = ImageSchema.readImages("hdfs://pierre:41234/cs455/images");

   // only if running locally, this will print the dataset schema to the console
   ds.printSchema();

   // shows top 20 entries in the schema and prints to console, again only prints if running locally
   ds.select(ds.col("image.width"), ds.col("image.height"), ds.col("image.mode")).show();

   ds.write().save("hdfs://pierre:41234/cs455/image-processing-output");

  }
}
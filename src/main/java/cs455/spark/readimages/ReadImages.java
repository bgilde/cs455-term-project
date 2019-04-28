package cs455.spark.readimages;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.*;
import org.apache.spark.ml.*;
import org.apache.spark.ml.image.*;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;
import java.io.File;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.awt.*;
import java.awt.image.ColorModel;
import java.awt.Rectangle;
import javax.imageio.ImageIO;

import org.opencv.core.Core; 
import org.opencv.core.Mat; 
import org.opencv.core.MatOfRect; 
import org.opencv.core.Point; 
import org.opencv.core.Rect; 
import org.opencv.core.Scalar; 
import org.opencv.imgcodecs.Imgcodecs; 
import org.opencv.imgproc.Imgproc; 
import org.opencv.objdetect.CascadeClassifier; 
import org.opencv.core.MatOfByte;
import org.opencv.core.CvType;

public class ReadImages {
  public static void main(String[] args) throws Exception {

   // Define a configuration to use to interact with Spark
   SparkConf conf = new SparkConf().setAppName("Image saver");

   // read images from hdfs
   Dataset<Row> ds = ImageSchema.readImages("hdfs://pierre:41234/cs455/images");

   // to local iterator will bring all of the partitions to one machine for processing but only one 
   // partition at a time to avoid memory issues
   Iterator<Row> iter = ds.toLocalIterator();

   while (iter.hasNext()) {
      Row row = iter.next();
      byte[] rawImageData = ImageSchema.getData(((Row)row.get(0)));
      nu.pattern.OpenCV.loadShared();
      System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

      CascadeClassifier faceDetector = new CascadeClassifier(); 
      faceDetector.load("/s/chopin/a/grad/bgilde/distributed-systems/spark/read-images/haarcascade_frontalface_alt.xml"); 

      // byte[] rawImageData = ImageSchema.getData(((Row)row.get(0)));

      Mat image = new Mat(480,854,CvType.CV_8UC3);
      image.put(0, 0, rawImageData);

      // Detecting faces 
      MatOfRect faceDetections = new MatOfRect(); 
      faceDetector.detectMultiScale(image, faceDetections); 

      // Creating a rectangular box showing faces detected 
      for (Rect rect : faceDetections.toArray()) 
      { 
         Imgproc.rectangle(image, new org.opencv.core.Point(rect.x, rect.y), 
         new org.opencv.core.Point(rect.x + rect.width, rect.y + rect.height), 
                                       new Scalar(0, 255, 0)); 
      } 

      Random r = new Random();
      int name = r.nextInt(500000);

      // Saving the output image 
      String filename = name + ".jpg"; 
      Imgcodecs.imwrite(filename, image);
   }
  }
}

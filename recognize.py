"""example-app.py"""
from pyspark import SparkContext
import face_recognition
from pyspark import SparkContext
from pyspark.ml.image import ImageSchema
import cv2
import numpy as np
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

if __name__ == '__main__':

   sc = SparkContext(appName="Example App")

   image_rdd = ImageSchema.readImages("hdfs://pierre:41234/cs455/images_large").rdd
   image_rdd = image_rdd.repartition(15)
   image_rdd.cache()

   '''
         note: look into broadcast variables for the face_cascade and embedder variable to avoid reading them from disk each iteration
   '''

   ## this function will create a 128 feature embedding for each face, now we need to train a neural network on top of this
   def f(x):
      face_cascade = cv2.CascadeClassifier('/s/chopin/a/grad/bgilde/distributed-systems/spark/python/haarcascade_frontalface_alt.xml')
      embedder = cv2.dnn.readNetFromTorch("/s/chopin/a/grad/bgilde/distributed-systems/spark/python/openface.nn4.small2.v1.t7")
      filename = x.image.origin.split('/')[-1]
      image = np.array(x.image.data)
      image = image.reshape((480,854,3))
      # cv2.imwrite("/s/chopin/a/grad/bgilde/distributed-systems/spark/python/new_image.jpg", image)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
         face = image[y:y+h,x:x+w,:]
         # cv2.imwrite("/s/chopin/a/grad/bgilde/distributed-systems/spark/python/test.jpg", face)
         faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                  (96, 96), (0, 0, 0), swapRB=True, crop=False)

         embedder.setInput(faceBlob)
         vec = embedder.forward()

         return (filename.strip('0123456789.png'), vec)

   # extract face embeddings
   feature_rdd = image_rdd.map(lambda x: f(x))

   # filter None values, meaning there was no face in the image
   feature_rdd = feature_rdd.filter(lambda x: x is not None)

   # test = feature_rdd.collect()

   print(feature_rdd.count())

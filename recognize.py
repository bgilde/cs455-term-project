from pyspark import SparkContext
import face_recognition
from pyspark import SparkContext
from pyspark.ml.image import ImageSchema
import cv2
import numpy as np
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == '__main__':

   sc = SparkContext(appName="Example App")

   image_rdd = ImageSchema.readImages("hdfs://pierre:41234/cs455/images_large").rdd
   image_rdd = image_rdd.repartition(15)
   image_rdd.cache()

   old_image_df = image_rdd.map(lambda line: Row(origin=line[0][0], data=line[0][5])).toDF()

   ## this function will create a 128 feature embedding for each face, now we need to train a neural network on top of this
   def f(old_image):
      face_cascade = cv2.CascadeClassifier('/s/chopin/a/grad/bgilde/distributed-systems/spark/python/haarcascade_frontalface_alt.xml')
      embedder = cv2.dnn.readNetFromTorch("/s/chopin/a/grad/bgilde/distributed-systems/spark/python/openface.nn4.small2.v1.t7")
      filename = old_image.image.origin.split('/')[-1]
      image = np.array(old_image.image.data)
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

         temp = filename.strip('0123456789.png')

         # return one hot encoded class label, feature vector and origin filename
         if 'Keith' in temp: return (0, vec, filename.strip('0123456789.png'), old_image.image.origin)
         else: return (1, vec, filename.strip('0123456789.png'), old_image.image.origin)

   # extract face embeddings
   feature_rdd = image_rdd.map(lambda x: f(x))

   # filter None values, meaning there was no face in the image
   feature_rdd_filtered = feature_rdd.filter(lambda x: x is not None).cache()

   test = feature_rdd_filtered.take(1)

   # print(test)

   # Map the RDD to a DF, DF have better performance in spark for machine learning applications
   feature_df_origin = feature_rdd_filtered.map(lambda line: Row(label=line[0], features=Vectors.dense(line[1][0].tolist()), origin=line[3])).toDF()

   feature_df = feature_df_origin.drop('origin')

   # Split the data into train and test sets
   train_data, test_data = feature_df.randomSplit([.8,.2],seed=1234)

   # get the test data with the origin so we can output the results later
   test_data_with_origin = feature_df_origin.join(test_data, feature_df_origin.features == test_data.features, 'inner')

   combined_test_data = test_data_with_origin.join(old_image_df, test_data_with_origin.origin == old_image_df.origin, 'inner')

   # specify layers for the neural network:
   # input layer of size 128 (features), two intermediate of size 5 and 4
   # and output of size 2 (classes)
   layers = [128, 5, 4, 2]

   # create the trainer and set its parameters
   trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

   # train the model
   model = trainer.fit(train_data)

   # compute accuracy on the test set
   result = model.transform(test_data)
   predictionAndLabels = result.select("prediction", "label")

   evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
   print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

   # Print out first 5 instances of `predictionAndLabel` 
   predictionAndLabels.show(10)


   def verify(old_image):
      face_cascade = cv2.CascadeClassifier('/s/chopin/a/grad/bgilde/distributed-systems/spark/python/haarcascade_frontalface_alt.xml')
      filename = old_image.origin.split('/')[-1]
      image = np.array(old_image.data)
      image = image.reshape((480,854,3))
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
         if old_image.label == 0: cv2.putText(image, "Keith Kirkland", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         else: cv2.putText(image, "Nora McInerny", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         face = image[y:y+h,x:x+w,:]
         cv2.imwrite("/s/chopin/a/grad/bgilde/distributed-systems/spark/python/" + filename, image)
         faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                  (96, 96), (0, 0, 0), swapRB=True, crop=False)

   test_set_rdd = combined_test_data.rdd

   test_set_rdd = test_set_rdd.map(lambda x: verify(x))

   test_set_rdd.count()

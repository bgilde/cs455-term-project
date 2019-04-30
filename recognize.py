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

   image_rdd = ImageSchema.readImages("hdfs://pierre:41234/cs455/combined_images").rdd
   image_rdd = image_rdd.repartition(20)
   image_rdd.cache()

   old_image_df = image_rdd.map(lambda line: Row(origin=line[0][0], data=line[0][5])).toDF()

   ## this function will create a 128 feature embedding for each face, now we need to train a neural network on top of this
   def f(old_image):
      # load the two pre-trained neural networks
      face_cascade = cv2.CascadeClassifier('/s/chopin/a/grad/bgilde/distributed-systems/spark/python/haarcascade_frontalface_alt.xml')
      embedder = cv2.dnn.readNetFromTorch("/s/chopin/a/grad/bgilde/distributed-systems/spark/python/openface.nn4.small2.v1.t7")
      filename = old_image.image.origin.split('/')[-1]
      image = np.array(old_image.image.data)
      image = image.reshape((480,854,3))
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # detect location of face in the image
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      # extract the 128 features from the face
      for (x,y,w,h) in faces:
         face = image[y:y+h,x:x+w,:]
         # blob from image will subtract the mean and normalize the image to reduce noise
         faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                  (96, 96), (0, 0, 0), swapRB=True, crop=False)

         embedder.setInput(faceBlob)
         vec = embedder.forward()

         # extract name to use as label
         temp = filename.strip('0123456789.png')

         # if 'Monique' in temp: return (0, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         # elif 'Greta' in temp: return (1, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         # else: return (2, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)

         # return one hot encoded class label, feature vector and origin filename
         if 'Amanda' in temp: return (0, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Ashweetha' in temp: return (1, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Carole' in temp: return (2, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Christoph' in temp: return (3, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Danielle-Lee' in temp: return (4, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Danielle-Moss' in temp: return (5, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Dina' in temp: return (6, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Douglas' in temp: return (7, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Elizabeth' in temp: return (8, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Emily' in temp: return (9, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Esha' in temp: return (10, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Eugenia' in temp: return (11, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Greta' in temp: return (12, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Hannah' in temp: return (13, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Julian' in temp: return (14, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Kekenya' in temp: return (15, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Katharine' in temp: return (16, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Kotchakorn' in temp: return (17, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Marjanvan' in temp: return (18, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Martin' in temp: return (19, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Matt' in temp: return (20, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Michael' in temp: return (21, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Monique' in temp: return (22, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Muhammed' in temp: return (23, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Nadjia' in temp: return (24, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Nicaila' in temp: return (25, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Olympia' in temp: return (26, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Phil' in temp: return (27, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'pj' in temp: return (28, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Priyanka' in temp: return (29, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Rana' in temp: return (30, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Samy' in temp: return (31, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Sarah' in temp: return (32, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Sean' in temp: return (33, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         elif 'Soraya' in temp: return (34, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)
         else: return (35, vec, filename.strip('0123456789.png'), old_image.image.origin, old_image.image.data)

   # extract face embeddings
   feature_rdd = image_rdd.map(lambda x: f(x))

   # filter None values, meaning there was no face in the image
   feature_rdd_filtered = feature_rdd.filter(lambda x: x is not None).cache()

   # Map the RDD to a DF, DF have better performance in spark for machine learning applications
   feature_df = feature_rdd_filtered.map(lambda line: Row(label=line[0], features=Vectors.dense(line[1][0].tolist()), origin=line[3], data = line[4])).toDF()

   # Split the data into train and test sets
   train_data, test_data = feature_df.randomSplit([.8,.2],seed=1234)

   # specify layers for the neural network:
   # input layer of size 128 (features), two intermediate of size 5 and 4
   # and output of size 3 (classes)
   layers = [128, 64, 64, 36]

   # class pyspark.ml.classification.MultilayerPerceptronClassifier(self, featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, tol=1e-6, seed=None, layers=None, blockSize=128, stepSize=0.03, solver="l-bfgs", initialWeights=None)[source]
   # create the trainer and set its parameters
   trainer = MultilayerPerceptronClassifier(maxIter=200, layers=layers, blockSize=128, seed=1234)

   # train the model
   model = trainer.fit(train_data)

   # compute accuracy on the test set
   result = model.transform(test_data)

   # result.printSchema()

   predictionAndLabels = result.select("prediction", "label")

   evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
   print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

   # Print out first 5 instances of `predictionAndLabel` 
   predictionAndLabels.show(10)

   result.printSchema()

   def verify(old_image):
      face_cascade = cv2.CascadeClassifier('/s/chopin/a/grad/bgilde/distributed-systems/spark/python/haarcascade_frontalface_alt.xml')
      # embedder = cv2.dnn.readNetFromTorch("/s/chopin/a/grad/bgilde/distributed-systems/spark/python/openface.nn4.small2.v1.t7")
      filename = old_image.origin.split('/')[-1]
      image = np.array(old_image.data)
      image = image.reshape((480,854,3))
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
         # check the prediction of the image, put corresponding label
         if old_image.prediction == 0: cv2.putText(image, "Amanda", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 1: cv2.putText(image, "Ashweetha", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 2: cv2.putText(image, "Carole", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 3: cv2.putText(image, "Christoph", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 4: cv2.putText(image, "Danielle-Lee", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 5: cv2.putText(image, "Danielle-Moss", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 6: cv2.putText(image, "Dina", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 7: cv2.putText(image, "Douglas", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 8: cv2.putText(image, "Elizabeth", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 9: cv2.putText(image, "Emily", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 10: cv2.putText(image, "Esha", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 11: cv2.putText(image, "Eugenia", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 12: cv2.putText(image, "Greta", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 13: cv2.putText(image, "Hannah", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 14: cv2.putText(image, "Julian", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 15: cv2.putText(image, "Kekenya", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 16: cv2.putText(image, "Katharine", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 17: cv2.putText(image, "Kotchakorn", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 18: cv2.putText(image, "Marjanvan", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 19: cv2.putText(image, "Martin", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 20: cv2.putText(image, "Matt", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 21: cv2.putText(image, "Michael", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 22: cv2.putText(image, "Monique", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 23: cv2.putText(image, "Muhammed", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 24: cv2.putText(image, "Nadjia", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 25: cv2.putText(image, "Nicaila", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 26: cv2.putText(image, "Olympia", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 27: cv2.putText(image, "Phil", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 28: cv2.putText(image, "pj", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 29: cv2.putText(image, "Priyanka", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 30: cv2.putText(image, "Rana", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 31: cv2.putText(image, "Samy", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 32: cv2.putText(image, "Sarah", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 33: cv2.putText(image, "Sean", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         elif old_image.prediction == 34: cv2.putText(image, "Soraya", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         else: cv2.putText(image, "Thomas", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
         cv2.imwrite("/s/chopin/a/grad/bgilde/distributed-systems/spark/python/" + filename, image)

   result_rdd = result.rdd

   result_rdd = result_rdd.map(lambda x: verify(x))

   result_rdd.count()

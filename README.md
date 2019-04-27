# cs455-term-project
## compilation instructions
A build.gradle file is included, so simply build with the command "gradle build". This command will produce a jar file called output.jar in the build/libs directory.

## running the program
Compiling will produce a jar and you simply have to submit the jar file to your spark cluster to run the program. Use the command "$SPARK_HOME/bin/spark-submit --master <spark://Spark_master_hostname:spark_master_port> --deploy-mode client --class cs455.spark.readimages.ReadImages --supervise ./build/libs/output.jar"

where <spark://Spark_master_hostname:spark_master_port> is replaced by whatever machine and port you set up spark on, for example for me this is "spark://pierre:21568". Also note that deploy mode is set to client so it runs locally on a single machine, this is the only way to see console output. To run on all nodes on your cluster, set deploy-mode to cluster instead of client.

In the program, you will have to change line 26:

  Dataset<Row> ds = ImageSchema.readImages("hdfs://pierre:41234/cs455/images");
  
and pass it an argument for whatever machine/port combo you have your hdfs set up on. Same thing with line 34:

  ds.write().save("hdfs://pierre:41234/cs455/image-processing-output");

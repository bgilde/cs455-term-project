pathToVideos = input("Path to folder with videos:\n")
if pathToVideos=="":
  pathToVideos = "./videos" 

import os
import cv2
for file in os.listdir(pathToVideos):
  if file.endswith(".mp4"):
    name = os.path.join(pathToVideos, file)
    print(name)

    cap = cv2.VideoCapture(name)
    frameCounter = 0

    while cap.isOpened() :#and limit > 0):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:    
        # Display the resulting frame
        #cv2.imshow('Frame',frame)
        outputName = "./images/"+file[:-4]+"_" + str(frameCounter) + ".png"
        print(outputName)
        cv2.imwrite(outputName, frame)

        frameCounter += 1
      # Break the loop if no frame returned
      else: break

    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

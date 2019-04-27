import cv2

cap = cv2.VideoCapture('./KeithKirkland_2018S-480p.mp4')

limit = 100

while(cap.isOpened() and limit > 0):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)

    cv2.imwrite("./" + str(limit) + ".png", frame)

    limit -= 1


 
  # Break the loop
  else: break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

import cv2
import numpy as np
import imutils

backsub = cv2.createBackgroundSubtractorMOG2() #background subtraction to isolate moving cars
capture = cv2.VideoCapture("/home/pi/Project_embed/Video Joiner190615115536.mp4")

frameSize = (600, 600)
areaFrame = frameSize[0] * frameSize[1]

car_cascade = cv2.CascadeClassifier('cars.xml')

counter = 0
x = 0
y = 0
sen = 0
minArea=1
lineCount = 250

if capture.isOpened():
    ret, frame = capture.read()
else:
    ret = False

while ret:
    ret, frame = capture.read()
    if ret==False:
        break

    frame = imutils.resize(frame, width=frameSize[0])
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,Gray = cv2.threshold(gray,40,100,cv2.THRESH_BINARY)
    fgmask = backsub.apply(frame, None, 0.01)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    erode=cv2.erode(fgmask,None,iterations=3)     #erosion to erase unwanted small contours
    moments=cv2.moments(erode,True)               #moments method applied
    area=moments['m00']
            
    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        if moments['m01'] >=minArea:
            x=int(moments['m10']/moments['m00'])
            y=int (moments['m01']/moments['m00'])
            if y < lineCount:
                sen=sen << 1
                print(sen)
            else:
                sen=(sen<<1)|1
                print ("sen = ")
                print (sen)
            sen=sen&0x03
            if sen == 1:
                counter=counter+1
                
        #cv2.circle(frame,(x,y),5,(0,0,255),-1) #center, radius, colour, -1=fill
    
    cv2.line(frame,(100,lineCount),(530,lineCount),(0,255,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'counter='+str(counter), (10,30),font,1, (255, 0, 0), 2)
    cv2.imshow("counter", frame)
    cv2.imshow("Gray", Gray)
    cv2.imshow("blacksup",fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()

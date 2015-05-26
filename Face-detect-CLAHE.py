import cv2
import sys

def main():
    faceCascade=cv2.CascadeClassifier("Cascades/front_alt.xml")
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    cap=cv2.VideoCapture()
    vidPath="rtsp://root:pass123@192.168.137.89:554/axis-media/media.amp"
    cap.open(vidPath)
    count =-1
    skip=5 #Minimum
    while(cap.isOpened()):
        ret,frame0=cap.read()
        count=count+1
        if count%skip==0:
            gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY) #scale down image
            gray1=clahe.apply(gray)
            #gray=cv2.equalizeHist(gray)
            faces= faceCascade.detectMultiScale(gray1,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
            for (x,y,w,h) in faces:
                instance=gray[x:x+w,y:y+h]
            
                cv2.rectangle(frame0,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('Video',frame0)
            cv2.imshow('Video2',gray1)
            if cv2.waitKey(100) & 0xFF== ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
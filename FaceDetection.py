import cv2
import mediapipe as mp

cap=cv2.VideoCapture("video3.mp4")

mpFaceDetection=mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection(0.15)#Burdaki sayı azaldık.a fazla detect yapar

mpDraw=mp.solutions.drawing_utils


while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results=faceDetection.process(imgRGB)

    if results.detections:
        for id,detection in enumerate(results.detections):
            bboxC=detection.location_data.relative_bounding_box

            h,w,_=img.shape
            bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
            cv2.rectangle(img,bbox,(0,255,255),2)


    cv2.imshow("img",img)
    k=cv2.waitKey(10) &0xFF
    if k==ord('q'):
        break
    
    


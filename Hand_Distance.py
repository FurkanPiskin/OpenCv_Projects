import cv2
import time
import numpy as np
import mediapipe as mp
import math

#4 ile 8

def drawLine(img,x1,y1,x2,y2):
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    distance=math.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

wCam,hCam=640,480
cap=cv2.VideoCapture(0)
mpHand=mp.solutions.hands
hands=mpHand.Hands()
mpDraw=mp.solutions.drawing_utils
handList=[]
param=0.5
doluluk=0
normalized_distance=0
cap.set(4,wCam)
cap.set(4,hCam)
pTime=0
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for id,handLms in enumerate(results.multi_hand_landmarks):
            mpDraw.draw_landmarks(img,handLms,mpHand.HAND_CONNECTIONS)

            h, w, _ = img.shape

            for id, lm in enumerate(handLms.landmark):
                # Her bir landmark'ın x ve y koordinatını hesapla
                cx, cy = int(lm.x * w), int(lm.y * h)
                handList.append([id, cx, cy])

                if id==4:
                    x1,y1=cx,cy
                    cv2.circle(img,(x1,y1),5,(0,0,255),2,cv2.FILLED)
                elif id==8:
                    x2,y2=cx,cy
                    cv2.circle(img,(x2,y2),5,(0,0,255),2,cv2.FILLED)

            if x1 is not None and x2 is not None:
                distance=drawLine(img,x1,y1,x2,y2)
                normalized_distance = np.interp(distance, (25, 290), (400, 150))
                 # Doluluk oranını ayarlama

                
                if y1==y2:
                    distance=0
                cv2.putText(img, str(int(distance)), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                #print(handList)
           
            
    cv2.rectangle(img, (50, 400),(85,int(normalized_distance)), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(img,(50,400),(85,150),(0,255,0))
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS:{int(fps)}',(40,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
    cv2.imshow("Img",img)
    k=cv2.waitKey(1) & 0xFF
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    
import cv2
import mediapipe as mp

# Kamerayı başlat
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Genişlik
cap.set(4, 480)  # Yükseklik

# MediaPipe el algılama modülü
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds=[4,8,12,16,20]

while True:
    success, img = cap.read()  # Kameradan görüntü al
    if not success:  # Eğer görüntü alınamazsa döngüyü kır
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Görüntüyü RGB'ye dönüştür
    results = hands.process(imgRGB)  # El algılama işlemi
    lmList=[]
    if results.multi_hand_landmarks:  # Eğer eller tespit edildiyse
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            
            for id,lm in enumerate(handLms.landmark):
                h,w,_=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                # işaret uc=8
               # if id==8:
               #     cv2.circle(img,(cx,cy),9,(255,0,0),cv2.FILLED)
               # if id==6:
                #    cv2.circle(img,(cx,cy),9,(0,0,255),cv2.FILLED)    
    if len(lmList)!=0:
        fingers=[]
        #bas parmak
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)     

        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0) 
        

        totalF=fingers.count(1)
        # print(totalF)
        cv2.putText(img,str(totalF),(30,125),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),8)  



    cv2.imshow("img", img)  # Görüntüyü göster
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında çık
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

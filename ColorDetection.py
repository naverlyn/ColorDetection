import cv2
import urllib.request
import numpy as np
 
def nothing(x):
    pass
 
url='http://192.168.1.10/cam-hi.jpg' #uri ngambil yang high resolution
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''
cv2.namedWindow("Window Deteksi Warna", cv2.WINDOW_AUTOSIZE)


#warna kode menggunakan BGR
l_h, l_s, l_v = 92, 57, 50
u_h, u_s, u_v = 142, 153, 178
i_h, i_s, i_v = 24, 0, 0 #new code test
 
while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgnp,-1)
    #_, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    i_b = np.array([i_h, i_s, i_v]) #new code test

    mask = cv2.inRange(hsv, l_b, u_b)
    draw = cv2.inRange(hsv, i_b, l_b) #draw dari i_b

    cnts, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    test, _ = cv2.findContours(draw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
    for c in cnts:
        area=cv2.contourArea(c)
        if area>2000:
            cv2.drawContours(frame,[c],-1,(255,0,0),3)
            M=cv2.moments(c)
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])
 
            #cv2.circle(frame,(cx,cy),3,(255,255,255),-1)
            cv2.putText(frame,"Tanah membutuhkan air",(cx-12, cy-12),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    
    for t in test:
        area=cv2.contourArea(t)
        if area>1500:
            cv2.drawContours(frame,[t], -1,(255, 0, 0), 3)
            M=cv2.moments(t)
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])
            cv2.circle(frame,(cx,cy),1,(255,255,255),-1)
            cv2.putText(frame,"warna apa",(cx-12, cy-12),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
 
    cv2.imshow("Window Deteksi Warna", frame)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break
    
 
cv2.destroyAllWindows()

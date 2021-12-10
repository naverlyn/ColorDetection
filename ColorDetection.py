import cv2
import urllib.request
import numpy as np
 
def nothing(x):
    pass
 
url='http://192.168.43.254/cam-hi.jpg' #uri ngambil yang high resolution
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''
cv2.namedWindow("Window Deteksi Warna", cv2.WINDOW_AUTOSIZE)


#warna kode menggunakan BGR
l_h, l_s, l_v = 92, 57, 50
u_h, u_s, u_v = 142, 153, 178
i_h, i_s, i_v = 24, 0, 0 #new code test
#========================================#
k_h, k_s, k_v = 57, 37, 26 #blue dark
m_h, m_s, m_v = 26, 18, 19 #keyboard
n_h, n_h, n_v = 159, 114, 93 #keyboard 2
 
while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgnp,-1)
    #_, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    i_b = np.array([i_h, i_s, i_v]) #new code test
    k_b = np.array([k_h, k_s, k_v]) #blue dark
    m_b = np.array([m_h, m_s, m_v]) #keyboard
    n_b = np.array([n_h, n_h, n_v]) #keyboard 2

    mask = cv2.inRange(hsv, l_b, u_b)
    draw = cv2.inRange(hsv, i_b, l_b) #draw dari i_b
    blue_dark = cv2.inRange(hsv, k_b, l_b) #detect blue dark
    keyboard = cv2.inRange(hsv, m_b, l_b) #detect keyboard
    keyboard2 = cv2.inRange(hsv, n_b, l_b) #detect keyboard 2


    cnts, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    test, _ = cv2.findContours(draw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    bd, _ = cv2.findContours(blue_dark,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    kb, _ = cv2.findContours(keyboard,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    kb2, _ = cv2.findContours(keyboard2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for b in bd:
        area=cv2.contourArea(b)
        if area>2000:
            cv2.drawContours(frame,[b],-1,(255,0,0),3)
            M=cv2.moments(b)
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])
 
            #cv2.circle(frame,(cx,cy),3,(255,255,255),-1)
            cv2.putText(frame,"Laptop",(cx-12, cy-12),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    for k in kb:
        area=cv2.contourArea(k)
        if area>1500:
            cv2.drawContours(frame,[k], -1,(255, 0, 0), 3)
            M=cv2.moments(k)
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])
            cv2.circle(frame,(cx,cy),1,(255,255,255),-1)
            cv2.putText(frame,"Keycaps",(cx-12, cy-12),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    for k2 in kb2:
        area=cv2.contourArea(k2)
        if area>1000:
            cv2.drawContours(frame,[k2], -1,(255, 0, 0), 3)
            M=cv2.moments(k2)
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])
            cv2.circle(frame,(cx,cy),1,(255,255,255),-1)
            cv2.putText(frame,"Keyboard",(cx-12, cy-12),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
 
    cv2.imshow("Window Deteksi Warna", frame)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break
    
 
cv2.destroyAllWindows()

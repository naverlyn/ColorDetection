import numpy as np
import cv2
import urllib.request
  
#url='http://192.168.43.254/cam-hi.jpg'     #URL dimasukkan kesini untuk request dari urllib
#webcam = cv2.VideoCapture('rtsp://192.168.43.1:8080/h264_ulaw.sdp')  
# Start
while(1):
    img_resp = urllib.request.urlopen(url)                          #tarik url ke img_resp untuk mengambil data input dari kamera
    imgnp = np.array(bytearray(img_resp.read()), dtype = np.uint8)  #ekstrak data menjadi data
    imageFrame = cv2.imdecode(imgnp, -1)                            #decode data menjadi image (videoStream)
    #_, imageFrame = webcam.read()
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)          #konversi BGR ke HSV
  
    # Set range for tanahSetengahKering dan 
    # define mask
    tanahSetengahKering_lower = np.array([17, 93, 67], np.uint8)            #tanah setengah kering
    tanahSetengahKering_upper = np.array([169, 183, 168], np.uint8)             #mask tanah setengah kering
    tanahSetengahKering_mask = cv2.inRange(hsvFrame, tanahSetengahKering_lower, tanahSetengahKering_upper)
  
    # Set range for tanahKering dan
    # define mask
    tanahKering_lower = np.array([17, 142, 81], np.uint8)          #tanah kering
    tanahKering_upper = np.array([195, 233, 185], np.uint8)          #mask tanah kering
    tanahKering_mask = cv2.inRange(hsvFrame, tanahKering_lower, tanahKering_upper)
  
    # Set range for tanahBasah dan
    # define mask
    tanahBasah_lower = np.array([17, 93, 67], np.uint8)                 #tanah basah
    tanahBasah_upper = np.array([195, 233, 185], np.uint8)             #mask tanah basah
    tanahBasah_mask = cv2.inRange(hsvFrame, tanahBasah_lower, tanahBasah_upper)
      
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")
      
    # For tanahSetengahKering color
    tanahSetengahKering_mask = cv2.dilate(tanahSetengahKering_mask, kernal)
    res_tanahSetengahKering = cv2.bitwise_and(imageFrame, imageFrame, 
                              mask = tanahSetengahKering_mask)
      
    # For tanahKering color
    tanahKering_mask = cv2.dilate(tanahKering_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask = tanahKering_mask)
      
    # For tanahBasah color
    tanahBasah_mask = cv2.dilate(tanahBasah_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = tanahBasah_mask)
   
    # buat contour untuk track tanahSetengahKering color
    contours, hierarchy = cv2.findContours(tanahSetengahKering_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (57, 56, 27), 2)
              
            cv2.putText(imageFrame, "Tanah St. Kering", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255))    
  
    # buat contour untuk track tanahKering color
    contours, hierarchy = cv2.findContours(tanahKering_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h),
                                       (0, 255, 0), 2)
              
            cv2.putText(imageFrame, "Tanah Kering", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0))
  
    # buat contour untuk track tanahBasah color
    contours, hierarchy = cv2.findContours(tanahBasah_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)
              
            cv2.putText(imageFrame, "Tanah Basah", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))
              
    # panggil program
    cv2.imshow("Window Deteksi Warna", imageFrame)
    #cv2.imshow("Mask Area", tanahKering_mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

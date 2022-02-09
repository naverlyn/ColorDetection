import cv2
import time
import logging
import numpy as np

webcam = cv2.VideoCapture('rtsp://192.168.1.7:8080/h264_pcm.sdp')
img_counter = 0
#webcam = cv2.VideoCapture(0)


def logTanahKering():
    logging.basicConfig(filename="log.txt",
                        filemode="a",
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%H:%M:%S')
    logging.info('found tanah Kering di lokasi: ' +
                 'Tanah ini harus diberi air!')


def logTanahSetengahKering():
    logging.basicConfig(filename="log.txt",
                        filemode="a",
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%H:%M:%S')
    logging.info('found tanah Setengah Kering di lokasi: ' +
                 'Tanah ini harus diberi sedikit air!')


def logTanahBasah():
    logging.basicConfig(filename="log.txt",
                        filemode="a",
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%H:%M:%S')
    logging.info('found tanah Basah di lokasi: ' +
                 'Tanah tidak perlu diberi air!')


if not webcam.isOpened():
    print("Gagal menyakalan kamera")
    exit()
while True:
    ret, imageFrame = webcam.read()
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    tanahSetengahKering_lower = np.array(
        [17, 93, 67], np.uint8)  # tanah setengah kering
    tanahSetengahKering_upper = np.array(
        [169, 183, 168], np.uint8)  # mask tanah setengah kering
    tanahSetengahKering_mask = cv2.inRange(
        hsvFrame, tanahSetengahKering_lower, tanahSetengahKering_upper)

    # Set range for tanahKering dan
    # define mask
    tanahKering_lower = np.array([17, 142, 81], np.uint8)  # tanah kering
    tanahKering_upper = np.array(
        [195, 233, 185], np.uint8)  # mask tanah kering
    tanahKering_mask = cv2.inRange(
        hsvFrame, tanahKering_lower, tanahKering_upper)

    # Set range for tanahBasah dan
    # define mask
    tanahBasah_lower = np.array([17, 93, 67], np.uint8)  # tanah basah
    tanahBasah_upper = np.array([195, 233, 185], np.uint8)  # mask tanah basah
    tanahBasah_mask = cv2.inRange(hsvFrame, tanahBasah_lower, tanahBasah_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For tanahSetengahKering color
    tanahSetengahKering_mask = cv2.dilate(tanahSetengahKering_mask, kernal)
    res_tanahSetengahKering = cv2.bitwise_and(imageFrame, imageFrame,
                                              mask=tanahSetengahKering_mask)

    # For tanahKering color
    tanahKering_mask = cv2.dilate(tanahKering_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=tanahKering_mask)

    # For tanahBasah color
    tanahBasah_mask = cv2.dilate(tanahBasah_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=tanahBasah_mask)

    # buat contour untuk track tanahSetengahKering color
    contours, hierarchy = cv2.findContours(tanahSetengahKering_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for picTanahSetengahKering, contour in enumerate(contours):
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

    for picTanahKering, contour in enumerate(contours):
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
    for picTanahBasah, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)

            cv2.putText(imageFrame, "Tanah Basah", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))

    if not ret:
        print("Tidak bisa mengambil frame")
        break
    img_name = "IMG_OPENCV_{}.png".format(img_counter)
    cv2.imwrite(img_name, imageFrame)
    img_counter += 1
    img = cv2.imread(img_name)
    cv2.imshow("w", img)
    if (picTanahKering <= picTanahBasah or picTanahKering < picTanahSetengahKering):
        logTanahKering()
    elif (picTanahKering >= picTanahKering or picTanahKering > picTanahBasah):
        logTanahSetengahKering()
    elif (picTanahKering != picTanahBasah or picTanahKering != picTanahSetengahKering):
        logTanahBasah()

    time.sleep(5)
    if cv2.waitKey(1) == ord('q'):
        break

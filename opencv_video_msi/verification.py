#!/usr/bin/env python3

DEVICE_NAME = "/dev/video10"

import cv2

cap = cv2.VideoCapture(DEVICE_NAME)

while True:
    _, img = cap.read()
    
    cv2.imshow(DEVICE_NAME, img)
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows

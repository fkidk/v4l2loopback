#!/usr/bin/env python3

import cv2

cap = cv2.VideoCapture("/dev/video10")

while True:
    _, img = cap.read()
    cv2.imshow('test', img)
    cv2.waitkey(1)

cv2.destroyAllWindowscd

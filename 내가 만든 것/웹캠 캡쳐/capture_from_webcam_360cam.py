import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import datetime
import time




def mouse_point(event, x, y, flags, params):
    global setCenter, Cx, Cy, circleShow, conform
    if event == cv2.EVENT_LBUTTONDOWN:
        Cx, Cy = x, y  # center of the "donut"
        setCenter = True
        circleShow = True

# build the mapping
def buildMap(R1, R2, Cx, Cy, Ws, Hs, Wd, Hd):
    map_x = np.zeros((int(Hd), int(Wd)), np.float32)
    map_y = np.zeros((int(Hd), int(Wd)), np.float32)
    for y in range(0, int(Hd - 1)):
        for x in range(0, int(Wd - 1)):
            r = (float(y) / float(Hd)) * (R2 - R1) + R1
            theta = (float(x) / float(Wd)) * 2.0 * np.pi
            xS = Cx + r * np.sin(theta)
            yS = Cy + r * np.cos(theta)
            map_x.itemset((y, x), int(xS))
            map_y.itemset((y, x), int(yS))

    return map_x, map_y


global map_x, map_y, hei, wei

setCenter = False
circleShow = False
conform = False

Cx, Cy = 0, 0
R1, R2 = 90, 185  # Inner/Outer donut radius

print("Set the Center Point")
cap = cv2.VideoCapture(-1)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_point)

if (cap.isOpened() == False):
    print("unable to read camera feed")

count = 0
maxs = 1000
sec = 1.5
while(True):
    ret, frame = cap.read()

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
                break

        if not setCenter and not conform and (Cx, Cy) == (0, 0):
            cv2.imshow("Frame", frame)

        elif setCenter and not conform:
            if circleShow is True:
                cv2.circle(frame, (Cx, Cy), 4, (0, 0, 255), -1)
                cv2.circle(frame, (Cx, Cy), R1, (0, 255, 255), 2)
                cv2.circle(frame, (Cx, Cy), R2, (255, 0, 255), 2)

            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            circleShow = False
            print('Center Point is setted')

            # our input and output image sizes
            Ws, Hs = frame.shape[:2]
            Wd = int(2.0 * ((R2 + R1) / 2) * np.pi)
            Hd = (R2 - R1)

            xmap, ymap = buildMap(R1, R2, Cx, Cy, Ws, Hs, Wd, Hd)
            print("MAP DONE!")
            conform = True

        if conform:
            # do an unwarping and show it to us
            panorama = cv2.remap(frame, xmap, ymap, cv2.INTER_LANCZOS4)
            break
    image = panorama
    
    print('Saved frame number : ' + str(int(cap.get(1))))
    cv2.imwrite("C:/Users/BK/Desktop/images/image%d.jpg" % count, image)
    print('Saved frame%d.jpg' % count)
    count += 1
    

    if count == maxs:
        break;
    

    time.sleep(sec)
    
    

cap.release()
cv2.destroyAllWindows()

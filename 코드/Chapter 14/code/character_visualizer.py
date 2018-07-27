import os
import sys

import cv2
import numpy as np

# Define the input file 
input_file = 'letter.data'  # 입력 받기

# Define the visualization parameters 
img_resize_factor = 12 # 파라미터 설정
start = 6
end = -1
height, width = 16, 8

# Iterate until the user presses the Esc key
with open(input_file, 'r') as f:
    for line in f.readlines():
        # Read the data
        data = np.array([255 * float(x) for x in line.split('\t')[start:end]]) #255개의 데이터를 가지고 온다. => 254개의 손글씨 이미지와 1개의 라벨

        # Reshape the data into a 2D image
        img = np.reshape(data, (height, width)) # 16*8 = 254 크기에 맞는 이미지로 재생성

        # Scale the image
        img_scaled = cv2.resize(img, None, fx=img_resize_factor, fy=img_resize_factor)

        # Display the image
        cv2.imshow('Image', img_scaled)

        # Check if the user pressed the Esc key
        c = cv2.waitKey()
        if c == 27:
            break

import cv2
import numpy as np

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier( # Haar cascade 파일을 사용
        'haar_cascade_files/haarcascade_frontalface_default.xml')

# Check if the cascade file has been loaded correctly
if face_cascade.empty(): # cascade 파일이 제대로 업로드 됬는지 확인
	raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object
cap = cv2.VideoCapture(0) # 웹캠 사용

# Define the scaling factor
scaling_factor = 0.5 # 이미지 크기 설정

# Iterate until the user hits the 'Esc' key
while True: # Esc가 나오기 전까지 실행
    # Capture the current frame
    _, frame = cap.read() # 현재프레임 가져오기

    # Resize the frame # 프레임 크기 재설정
    frame = cv2.resize(frame, None, 
            fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

    # Convert to grayscale # 회색으로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale image # 회색 이미지에서 얼굴 탐색
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the face # 얼굴에 사각형 그리기
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    # Display the output
    cv2.imshow('Face Detector', frame)

    # Check if the user hit the 'Esc' key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()

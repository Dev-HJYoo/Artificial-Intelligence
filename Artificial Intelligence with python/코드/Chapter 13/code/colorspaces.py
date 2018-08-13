import cv2
import numpy as np

# Define a function to get the current frame from the webcam
def get_frame(cap, scaling_factor): # 현재 프레임을 가지고 오는 것.
    # Read the current frame from the video capture object
    _, frame = cap.read()

    # Resize the image # 크기 조정
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame

if __name__=='__main__':
    # Define the video capture object  # 캠에서 정보 가져오기
    cap = cv2.VideoCapture(0) 

    # Define the scaling factor for the images # 크기
    scaling_factor = 0.5

    # Keep reading the frames from the webcam 
    # until the user hits the 'Esc' key
    while True: # Esc키가 나오기 전까지 게속 한다.
        # Grab the current frame
        frame = get_frame(cap, scaling_factor)  # 현재 프레임을 가지고 온다.

        # Convert the image to HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # RGB -> HSV로 변경

        # Define range of skin color in HSV # HSV내의 피부색깔 범위
        lower = np.array([0, 70, 60])
        upper = np.array([50, 150, 255])

        # Threshold the HSV image to get only skin color # HSV 이미지에서 피부색깔범위 만큼 threshold 설정
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND between the mask and original image # 현재 프레임들에서 threshold만큼 and 연
        img_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)

        # Run median blurring # medianBlur 기법으로 조금 더 잘보이게 함
        img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)
        
        # Display the input and output # RGB 모드와 HSV 모드
        cv2.imshow('Input', frame)
        cv2.imshow('Output', img_median_blurred)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(5) 
        if c == 27:
            break

    # Close all the windows
    cv2.destroyAllWindows()

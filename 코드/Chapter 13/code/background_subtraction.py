import cv2
import numpy as np

# Define a function to get the current frame from the webcam
def get_frame(cap, scaling_factor): # 프레임을 가져 오는 곳
    # Read the current frame from the video capture object
    _, frame = cap.read()

    # Resize the image # 크기 재조정
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame

if __name__=='__main__':
    # Define the video capture object
    cap = cv2.VideoCapture(0) # 웹캠 사용

    # Define the background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2() # background subtractor 객체 생성
     
    # Define the number of previous frames to use to learn. 
    # This factor controls the learning rate of the algorithm. 
    # The learning rate refers to the rate at which your model 
    # will learn about the background. Higher value for 
    # ‘history’ indicates a slower learning rate. You can 
    # play with this parameter to see how it affects the output.
    history = 1000# 학습 시킬 전 프레임의 갯수 ( 즉, 학습량)

    # Define the learning rate
    learning_rate = 1.0/history # 학습 비율

    # Keep reading the frames from the webcam 
    # until the user hits the 'Esc' key
    while True: # Esc 키 누르면 종료
        # Grab the current frame
        frame = get_frame(cap, 0.5) # 현재 프레임 가져오기

        # Compute the mask 
        mask = bg_subtractor.apply(frame, learningRate=learning_rate) # background 비교하기

        # Convert grayscale image to RGB color image
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 회색으로 바꾸기 

        # Display the images
        cv2.imshow('Input', frame)
        cv2.imshow('Output', mask & frame)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(10)
        if c == 27:
            break

    # Release the video capture object
    cap.release() # 캡쳐 해
    
    # Close all the windows
    cv2.destroyAllWindows()

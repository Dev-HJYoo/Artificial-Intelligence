import cv2

# Compute the frame differences
def frame_diff(prev_frame, cur_frame, next_frame): # 프레임의 다른 점을 계산하는 함수
    # Difference between the current frame and the next frame  # 현재 프레임과 다음 프레임 사이의 차이점
    diff_frames_1 = cv2.absdiff(next_frame, cur_frame)

    # Difference between the current frame and the previous frame # 전 프레임과 현재 프레임 사이의 차이점
    diff_frames_2 = cv2.absdiff(cur_frame, prev_frame)

    return cv2.bitwise_xor(diff_frames_1, diff_frames_2) # bit AND 연산( 둘다 1 이면 1 ) 로 두 프레임의 차이점을 리턴시킴.

# Define a function to get the current frame from the webcam 
def get_frame(cap, scaling_factor): # 웹캠을 통해서 프레임을 가지고 온다. 
    # Read the current frame from the video capture object
    _, frame = cap.read()

    # Resize the image   # 원하는 크기로 재설정
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Convert to grayscale # 회색으로 덮기
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return gray 

if __name__=='__main__':
    # Define the video capture object  # 웹캠 불러 오기
    cap = cv2.VideoCapture(0)

    # Define the scaling factor for the images # 이미지 크기
    scaling_factor = 0.5
    
    # Grab the current frame # 전 프레임
    prev_frame = get_frame(cap, scaling_factor) 

    # Grab the next frame # 현재 프레임
    cur_frame = get_frame(cap, scaling_factor) 

    # Grab the frame after that # 후 프레임
    next_frame = get_frame(cap, scaling_factor) 

    # Keep reading the frames from the webcam 
    # until the user hits the 'Esc' key
    while True: # Esc 키가 나올때 까지 계속 프레임을 가지고 온다.
        # Display the frame difference
        cv2.imshow('Object Movement', frame_diff(prev_frame, 
                cur_frame, next_frame))

        # Update the variables # 프레임 업데이트 
        prev_frame = cur_frame
        cur_frame = next_frame 

        # Grab the next frame # 후 프레임만 업데이트 시키면 된다.
        next_frame = get_frame(cap, scaling_factor)

        # Check if the user hit the 'Esc' key
        key = cv2.waitKey(10)
        if key == 27:
            break

    # Close all the windows
    cv2.destroyAllWindows()

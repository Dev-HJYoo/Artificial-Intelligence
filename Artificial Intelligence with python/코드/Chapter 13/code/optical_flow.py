import cv2
import numpy as np

# Define a function to track the object
def start_tracking(): # 트레킹 시작 함수
    # Initialize the video capture object
    cap = cv2.VideoCapture(0) # 웹캠 사용

    # Define the scaling factor for the frames
    scaling_factor = 0.5 # 크기 설정

    # Number of frames to track
    num_frames_to_track = 5 # 트레킹 하는 프레임 갯수

    # Skipping factor # 스킵하는 프레임 갯수
    num_frames_jump = 2

    # Initialize variables # 초기화
    tracking_paths = []
    frame_index = 0

    # Define tracking parameters # 트레킹 파라미터 선언
    tracking_params = dict(winSize  = (11, 11), maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                10, 0.03))

    # Iterate until the user hits the 'Esc' key
    while True: # Esc키가 나오기 전까지 실행
        # Capture the current frame
        _, frame = cap.read() # 현재 프레임 가져오기

        # Resize the frame # 크기 재설정
        frame = cv2.resize(frame, None, fx=scaling_factor, 
                fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Convert to grayscale # gray색깔로 바꾸기
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a copy of the frame # 현재 프레임 카피
        output_img = frame.copy()

        if len(tracking_paths) > 0: # 트레킹 요소가 있다면
            # Get images
            prev_img, current_img = prev_gray, frame_gray # 전 과 현재 프레임 설정

            # Organize the feature points # 특징 지점 설정
            feature_points_0 = np.float32([tp[-1] for tp in \
                    tracking_paths]).reshape(-1, 1, 2)

            # Compute optical flow # optical flow 계산
            feature_points_1, _, _ = cv2.calcOpticalFlowPyrLK(
                    prev_img, current_img, feature_points_0, 
                    None, **tracking_params)

            # Compute reverse optical flow # 역optical flow 계산
            feature_points_0_rev, _, _ = cv2.calcOpticalFlowPyrLK(
                    current_img, prev_img, feature_points_1, 
                    None, **tracking_params)

            # Compute the difference between forward and  # optical flow 와 역optical flow 다른점 계산
            # reverse optical flow
            diff_feature_points = abs(feature_points_0 - \
                    feature_points_0_rev).reshape(-1, 2).max(-1)

            # Extract the good points # 좋은 좌표 추출
            good_points = diff_feature_points < 1

            # Initialize variable # 트레킹 길 초기화
            new_tracking_paths = []

            # Iterate through all the good feature points  # 특징 좌표를 모두 실행
            for tp, (x, y), good_points_flag in zip(tracking_paths, 
                        feature_points_1.reshape(-1, 2), good_points):
                # If the flag is not true, then continue
                if not good_points_flag: # flag가 false이면 계속 실행
                    continue

                # Append the X and Y coordinates and check if
                # its length greater than the threshold
                tp.append((x, y)) # flag가 true이므로 추가한다. ( 더 좋은 결과가 나왔을 경우 )
                if len(tp) > num_frames_to_track:
                    del tp[0]

                new_tracking_paths.append(tp)

                # Draw a circle around the feature points
                cv2.circle(output_img, (x, y), 3, (0, 255, 0), -1)

            # Update the tracking paths  # 트레킹 패스 업데이트
            tracking_paths = new_tracking_paths

            # Draw lines # 트레킹 패스로 선 끗기
            cv2.polylines(output_img, [np.int32(tp) for tp in \
                    tracking_paths], False, (0, 150, 0))

        # Go into this 'if' condition after skipping the 
        # right number of frames
        if not frame_index % num_frames_jump: # 트레킹 요소가 없을 경우
            # Create a mask and draw the circles # 특징 좌표 및 원 생성
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
                cv2.circle(mask, (x, y), 6, 0, -1)

            # Compute good features to track # 트레킹 특징 계산
            feature_points = cv2.goodFeaturesToTrack(frame_gray, 
                    mask = mask, maxCorners = 500, qualityLevel = 0.3, 
                    minDistance = 7, blockSize = 7) 

            # Check if feature points exist. If so, append them
            # to the tracking paths
            if feature_points is not None: # 트레킹 패스 추가
                for x, y in np.float32(feature_points).reshape(-1, 2):
                    tracking_paths.append([(x, y)])

        # Update variables # 변수 업데이트
        frame_index += 1
        prev_gray = frame_gray

        # Display output # 출력
        cv2.imshow('Optical Flow', output_img)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(1)
        if c == 27:
            break

if __name__ == '__main__':
	# Start the tracker
    start_tracking()

    # Close all the windows
    cv2.destroyAllWindows()


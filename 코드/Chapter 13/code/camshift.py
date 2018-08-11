import cv2
import numpy as np

# Define a class to handle object tracking related functionality
class ObjectTracker(object): # 기능적으로 관련있는 물체를 tacking하기 위한 클래스
    def __init__(self, scaling_factor=0.5):
        # Initialize the video capture object
        self.cap = cv2.VideoCapture(0) # 웹캠 사용하기

        # Capture the frame from the webcam
        _, self.frame = self.cap.read() # 프레임 읽어 오기

        # Scaling factor for the captured frame
        self.scaling_factor = scaling_factor # 크기 설정 

        # Resize the frame # 크기 재설정
        self.frame = cv2.resize(self.frame, None, 
                fx=self.scaling_factor, fy=self.scaling_factor, 
                interpolation=cv2.INTER_AREA)

        # Create a window to display the frame # 프레임이름 설정
        cv2.namedWindow('Object Tracker')

        # Set the mouse callback function to track the mouse
        cv2.setMouseCallback('Object Tracker', self.mouse_event)

        # Initialize variable related to rectangular region selection # bounding box 초기화
        self.selection = None

        # Initialize variable related to starting position  # 시작점 초기화
        self.drag_start = None

        # Initialize variable related to the state of tracking  # 트레킹 상태 초기화
        self.tracking_state = 0

    # Define a method to track the mouse events
    def mouse_event(self, event, x, y, flags, param): # 마우스 event에 대한 함수
        # Convert x and y coordinates into 16-bit numpy integers
        x, y = np.int16([x, y])  # x와 y 좌표를 16비트로 설정

        # Check if a mouse button down event has occurred
        if event == cv2.EVENT_LBUTTONDOWN: # 마우스를 클릭할 경우
            self.drag_start = (x, y)
            self.tracking_state = 0

        # Check if the user has started selecting the region
        if self.drag_start: # 물체를 선택한 경우
            if flags & cv2.EVENT_FLAG_LBUTTON: # 버튼을 클릭함
                # Extract the dimensions of the frame
                h, w = self.frame.shape[:2] # 프레임 높이와 넓이 설정

                # Get the initial position
                xi, yi = self.drag_start # 초기 좌표

                # Get the max and min values # 사각형의 최대값과 최소 값 설정.
                x0, y0 = np.maximum(0, np.minimum([xi, yi], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xi, yi], [x, y]))

                # Reset the selection variable # 선택 초기화
                self.selection = None

                # Finalize the rectangular selection
                if x1-x0 > 0 and y1-y0 > 0: # 최종적으로 사각형 생성
                    self.selection = (x0, y0, x1, y1)

            else:
                # If the selection is done, start tracking   # 트레킹 시작
                self.drag_start = None
                if self.selection is not None:
                    self.tracking_state = 1

    # Method to start tracking the object
    def start_tracking(self):
        # Iterate until the user presses the Esc key
        while True: # Esc키를 누르기 전까지 재생
            # Capture the frame from webcam
            _, self.frame = self.cap.read() # 프레임 가져오기
            
            # Resize the input frame
            self.frame = cv2.resize(self.frame, None,  # 크기 재조성
                    fx=self.scaling_factor, fy=self.scaling_factor, 
                    interpolation=cv2.INTER_AREA)

            # Create a copy of the frame
            vis = self.frame.copy() # 프레임 카피

            # Convert the frame to HSV colorspace # HSV 컬러로 변경
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            # Create the mask based on predefined thresholds # thresholds 재설정
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), 
                        np.array((180., 255., 255.)))

            # Check if the user has selected the region
            if self.selection: # 구역을 설정한 경우
                # Extract the coordinates of the selected rectangle
                x0, y0, x1, y1 = self.selection # 좌표

                # Extract the tracking window # 트레킹 하는거 보여주기
                self.track_window = (x0, y0, x1-x0, y1-y0)

                # Extract the regions of interest  # 구역의 특징 추출
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                # Compute the histogram of the region of  # 막대그래프로 계산하기 
                # interest in the HSV image using the mask
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, 
                        [16], [0, 180] )

                # Normalize and reshape the histogram # 표준화 및 재형성
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                self.hist = hist.reshape(-1)

                # Extract the region of interest from the frame # 프레임에서 선택한 구역 추출
                vis_roi = vis[y0:y1, x0:x1]

                # Compute the image negative (for display only) # 이미지 음영 계산하기
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            # Check if the system in the "tracking" mode # 트레킹 모드일 경우
            if self.tracking_state == 1:
                # Reset the selection variable
                self.selection = None # 선택부분은 안한다. 
                
                # Compute the histogram back projection
                hsv_backproj = cv2.calcBackProject([hsv], [0],  # back projection 막대그래프 계산
                        self.hist, [0, 180], 1)

                # Compute bitwise AND between histogram 
                # backprojection and the mask # bit-and로 계산
                hsv_backproj &= mask

                # Define termination criteria for the tracker # 강제 종료 부분
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                        10, 1)

                # Apply CAMShift on 'hsv_backproj' # hsv_backproj를 적용시킨다.
                track_box, self.track_window = cv2.CamShift(hsv_backproj, 
                        self.track_window, term_crit)

                # Draw an ellipse around the object # bounding box 설정
                cv2.ellipse(vis, track_box, (0, 255, 0), 2)

            # Show the output live video # 영상 출
            cv2.imshow('Object Tracker', vis)

            # Stop if the user hits the 'Esc' key
            c = cv2.waitKey(5)
            if c == 27:
                break

        # Close all the windows
        cv2.destroyAllWindows()

if __name__ == '__main__':
	# Start the tracker
    ObjectTracker().start_tracking()


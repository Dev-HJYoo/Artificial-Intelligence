import cv2
import time

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("unable to read camera feed")

count = 0
maxs = 20
sec = 2
while(True):
    ret, image = cap.read()
    
    print('Saved frame number : ' + str(int(cap.get(1))))
    cv2.imwrite("../images/h%d.jpg" % count, image)
    print('Saved frame%d.jpg' % count)
    count += 1
    

    if count == maxs:
        break;
    time.sleep(sec)
    
    

cap.release()
cv2.destroyAllWindows()

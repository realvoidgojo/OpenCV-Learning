import cv2
import time
import modules.poseTrackingModule as PTM
import numpy as np

cap = cv2.VideoCapture("./clips/gym.mp4")
detector = PTM.poseDetector()

pTime = 0
count = 0
dir = 0  # 0 = down, 1 = up
percentBar = 400
percent = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        
        angle1 = detector.findAngle(img, 11, 13, 15, draw=True)
        angle2 = detector.findAngle(img, 12, 14, 16, draw=True) 
        # Average angle
        angle = (angle1 + angle2) / 2

        
        percent = np.interp(angle, (160, 45), (100, 0))          
        percentBar = np.interp(angle, (160, 45), (150, 400))     

        # Rep counting logic
        if percent == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if percent == 0:
            if dir == 1:
                count += 0.5
                dir = 0

        
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)  
        cv2.rectangle(img, (50, int(percentBar)), (85, 400), (0, 255, 0), cv2.FILLED)

        
        cv2.putText(img, f"{int(percent)}%", (40, 130),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

       
        cv2.putText(img, f"Reps: {int(count)}", (30, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 2)

   
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-5)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (30, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Bicep Curl Counter", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

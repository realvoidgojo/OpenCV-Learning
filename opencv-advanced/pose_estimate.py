import cv2
import time
import modules.poseTrackingModule as PM

def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(r"./../clips/yoga.mp4")
    detector = PM.poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        # print(lmList)
        if len(lmList) != 0:
            angle = detector.findAngle(img, 26, 24, 12)

            # print(f"Right Elbow Angle: {angle:.2f}")
            # Visual indicator (already part of findAngle)
            cv2.circle(img,(lmList[14][1], lmList[14][2]), 15, (0, 255, 0), cv2.FILLED)
            # find pose
            cv2.circle(img,(lmList[14][1],lmList[14][2]),15,(0,255,0),cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)

        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()


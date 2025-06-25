import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, modelcomplexity=1, smooth=True, 
                 detectCon=0.5,trackingCon=0.5):
        
        self.mode = mode
        self.modelcomplexity = modelcomplexity
        self.smooth = smooth
        self.enable_seg=False
        self.smooth_seg=True
        self.detectCon = detectCon
        self.trackingCon = trackingCon
        
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose( self.mode, self.modelcomplexity, self.smooth, 
                 self.enable_seg, self.smooth_seg, self.detectCon,self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                          self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self,img,draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape

                cx,cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self,img,p1,p2,p3,draw=True):
  
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3-y2 , x3-x2)-
                             math.atan2(y1-y2 , x1-x2))
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x3,y3),(x2,y2),(255,255,255),3)
            cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x1,y1),15,(0,0,255),2)
            cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(0,0,255),2)
            cv2.circle(img,(x3,y3),15,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x3,y3),15,(0,0,255),2)
            cv2.putText(img,str(int(angle)),(x2-50,y2+50),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        
        return angle

def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(r"./../clips/yoga.mp4")
    detector = poseDetector()

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
            

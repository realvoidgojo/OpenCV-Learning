import cv2
import mediapipe as mp
import time
from screeninfo import get_monitors

class FaceDetection():
    def __init__(self,minCon=0.5,model=0):
        self.minCon = minCon
        self.model = model
        self.mpFaceDetections = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDectections = self.mpFaceDetections.FaceDetection(self.minCon,self.model)

    def findFace(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.faceDectections.process(imgRGB)
        if results.detections:
            if draw:
                for id,detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih,iw,ic = img.shape
                    bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
                           int(bboxC.width * iw) , int(bboxC.height * ih)
                    cv2.rectangle(img,bbox,(255,0,255),2)
                    cv2.putText(img,f"{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),2)
        return img     


def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(r'../clips/face.mp4')
    monitor = get_monitors()[0]
    detector = FaceDetection()
    scrWidth, scrHeight = monitor.width, monitor.height

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img  = detector.findFace(img)
        h,w,_ = img.shape
        scale = min(scrWidth / w*0.5 , scrHeight /h*0.5) 
        resized_img = cv2.resize(img,(int(w*scale), int(h*scale)))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(resized_img,f"FPS: {int(fps)}",(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        cv2.imshow("img",resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break       

if __name__ == "__main__":
    main()
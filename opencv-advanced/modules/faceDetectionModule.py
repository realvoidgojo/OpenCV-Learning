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
        self.results = self.faceDectections.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih,iw,ic = img.shape
                    bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
                           int(bboxC.width * iw) , int(bboxC.height * ih)
                    bboxs.append([id,bbox,detection.score])
            if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.putText(img,f"{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),2)
        return img,bboxs
    
    def fancyDraw(self,img,bbox,l=30,t=5,rt=1):
         x,y,w,h = bbox
         x1 , y1 = x+w,y+h
         cv2.rectangle(img,bbox,(255,0,255),rt)
         # top left x,y
         cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
         cv2.line(img,(x,y),(x,y+l),(255,0,255),t)
         #top right x1,y
         cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t)
         cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)
         #bottom left
         cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t)
         cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t)
         #bottom right
         cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),t)
         cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),t)
         return img

def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)
    monitor = get_monitors()[0]
    detector = FaceDetection()
    scrWidth, scrHeight = monitor.width, monitor.height

    while True:
        success, img = cap.read()
        img,bboxs  = detector.findFace(img)
        print(bboxs)
        cTime = time.time() 
        fps = 1 / (cTime - pTime)
        pTime = cTime

        h,w,_ = img.shape
        scale = min(scrWidth / w*0.5 , scrHeight /h*0.5) 
        resized_img = cv2.resize(img,(int(w*scale), int(h*scale)))

        cv2.putText(resized_img,f"FPS: {int(fps)}",(30,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        cv2.imshow("img",resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break       

if __name__ == "__main__":
    main()
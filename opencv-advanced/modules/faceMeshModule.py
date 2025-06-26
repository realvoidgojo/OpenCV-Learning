import cv2
import mediapipe as mp
import time 

class FaceMeshDetector():
    def __init__(self,staticMode=False,maxFaces=2,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode = self.staticMode, 
            max_num_faces = self.maxFaces,
            min_detection_confidence = self.minDetectionCon,
            min_tracking_confidence = self.minTrackCon,
        )

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1,color=(255,0,255))

    def findFaceMesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec,self.drawSpec
                    )
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    ih,iw, ic = img.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)
                    face.append([x,y])
        return img,faces

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success,img = cap.read()
        if not success:
            break
        img,faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(faces[0])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f"FPS: {int(fps)}",(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)
        cv2.imshow("img",img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
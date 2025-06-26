import cv2
import mediapipe as mp
import time

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    mpFaceMesh = mp.solutions.face_mesh
    mpDraw = mp.solutions.drawing_utils
    faceMesh = mpFaceMesh.FaceMesh( 
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1,color=(0,255,0))

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_TESSELATION, landmark_drawing_spec=drawSpec,connection_drawing_spec=drawSpec)
                for id,lm in enumerate(faceLms.landmark):
                    ih,iw,ic = img.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)
                    print(id,x,y)
                    
    
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 0, 255), 2)

        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2

face_cascade = cv2.CascadeClassifier("haar_face.xml")

cap = cv2.VideoCapture(0)

tracker = None
tracking = None

while True:
	ret,frame = cap.read()
	if not ret:
		break
	if not tracking:
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 5,minSize=(50,50))

		if len(faces)>0:
			x,y,w,h = faces[0]
			tracker = cv2.TrackerCSRT_create()
			tracker.init(frame,(x,y,w,h))
			tracking = True
	else:
		success,bbox = tracker.update(frame)
		if success:
			x,y,w,h = [int(v) for v in bbox]
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.putText(frame, "Tracking Face", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
		else:
			tracker = None 
			tracking = False


	cv2.imshow("tracking", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q") or key == 27:
		break
	elif key == ord("r"):
		tracking = False
		tracker = None
		
cap.release()
cv2.destroyAllWindows()
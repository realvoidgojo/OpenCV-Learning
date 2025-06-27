import cv2
print("OpenCV version:", cv2.__version__)
cv2.VideoCapture(0)
def main():
    cap = cv2.VideoCapture("http://172.16.125.225:8080/video")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import pyautogui
import math


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

click_threshold = 40  
is_clicking = False
click_cooldown = 0

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1) 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            
            index_finger = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            x = int(index_finger.x * frame.shape[1])
            y = int(index_finger.y * frame.shape[0])

            
            screen_x = screen_width / frame.shape[1] * x
            screen_y = screen_height / frame.shape[0] * y
            pyautogui.moveTo(screen_x, screen_y)

            
            distance = calculate_distance(thumb_tip, index_finger) * frame.shape[1]
            
            
            if distance < click_threshold and not is_clicking and click_cooldown == 0:
                pyautogui.click()
                is_clicking = True
                click_cooldown = 10  
                print("CLICK GESTURE DETECTED - Pinch motion performed!")
                
            elif distance > click_threshold:
                is_clicking = False
                
           
            if click_cooldown > 0:
                click_cooldown -= 1

            
            cursor_color = (0, 0, 255) if is_clicking else (0, 255, 255)  # Red when clicking, yellow otherwise
            cv2.circle(frame, (x, y), 10, cursor_color, -1)
            
           
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])
            cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 0), -1)
            
            
            line_color = (0, 255, 0) if distance > click_threshold else (0, 0, 255)
            cv2.line(frame, (thumb_x, thumb_y), (x, y), line_color, 2)
            
           
            cv2.putText(frame, f"Distance: {int(distance)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            status = "CLICKING" if is_clicking else "READY"
            cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Tracking Mouse", frame)
    if cv2.waitKey(1) == 27:  
        break

cap.release()
cv2.destroyAllWindows()
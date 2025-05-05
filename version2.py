import cv2
import mediapipe as mp
import pyautogui
import time

# Face & emotion cascades
face_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Counters & flags
no_face_counter    = 0
EYE_CLOSED_COUNTER = 0
SMILE_COOLDOWN     = 3      # seconds between auto-likes
last_smile_time    = 0
paused_by_face     = False  # track if we auto-paused

# MediaPipe hands setup
mp_hands     = mp.solutions.hands
hands        = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw      = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_history  = []
HISTORY_LENGTH   = 5
prev_gesture     = None
last_action_time = 0
cooldown         = 1.0   # seconds between gesture actions

def detect_gesture(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(1 if landmarks[tips_ids[0]].x < landmarks[tips_ids[0]-1].x else 0)
    for i in range(1,5):
        fingers.append(1 if landmarks[tips_ids[i]].y < landmarks[tips_ids[i]-2].y else 0)

    if   fingers == [0,0,0,0,0]: return "fist"
    elif fingers == [1,1,1,1,1]: return "palm"
    elif fingers == [0,1,0,0,0]: return "point"
    elif fingers == [0,1,1,0,0]: return "volume_up"
    elif fingers == [0,1,0,0,1]: return "volume_down"
    elif fingers == [1,1,1,0,0]: return "fullscreen"
    return "unknown"

def perform_action(gesture):
    if gesture == "palm":
        pyautogui.press("space")
    elif gesture == "fist":
        pyautogui.press("esc")
    elif gesture == "point":
        pyautogui.press("right")
    elif gesture == "volume_up":
        pyautogui.press("volumeup")
    elif gesture == "volume_down":
        pyautogui.press("volumedown")
    elif gesture == "fullscreen":
        pyautogui.press("f")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Face / Eye / Smile detection
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        no_face_counter += 1
        if no_face_counter > 20:
            pyautogui.press("space")
            paused_by_face = True
            no_face_counter = 0
    else:
        no_face_counter = 0
        face_eyes_open = False

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            if len(eyes) == 0:
                EYE_CLOSED_COUNTER += 1
                if EYE_CLOSED_COUNTER > 20:
                    pyautogui.press("space")
                    paused_by_face = True
                    EYE_CLOSED_COUNTER = 0
            else:
                face_eyes_open = True
                EYE_CLOSED_COUNTER = 0

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
            if len(smiles) > 0 and (time.time() - last_smile_time) > SMILE_COOLDOWN:
                pyautogui.press("l")
                last_smile_time = time.time()

        if paused_by_face and face_eyes_open:
            pyautogui.press("space")
            paused_by_face = False

    # Hand-gesture detection
    current_gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_gesture = detect_gesture(hand_landmarks.landmark)

            gesture_history.append(current_gesture)
            if len(gesture_history) > HISTORY_LENGTH:
                gesture_history.pop(0)

            if gesture_history.count(current_gesture) >= 4:
                if current_gesture != prev_gesture and (time.time() - last_action_time) > cooldown:
                    perform_action(current_gesture)
                    prev_gesture     = current_gesture
                    last_action_time = time.time()

            cv2.putText(img, f'Gesture: {current_gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture & Smart Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

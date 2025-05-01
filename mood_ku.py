import cv2
import mediapipe as mp
import random
import os
import numpy as np

# Inisialisasi Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Load emoji
emoji_folder = "emojis"  # Ganti dengan folder tempat menyimpan emoji PNG
emoji_files = [os.path.join(emoji_folder, f) for f in os.listdir(emoji_folder) if f.endswith(".png")]
if not emoji_files:
    raise ValueError("Tidak ada file PNG dalam folder emojis")

# Variabel untuk tracking emoji
current_emoji = random.choice(emoji_files)
change_emoji = True
scale_factor = 1.7  # Skala ukuran emoji lebih besar

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi wajah
    face_results = face_detection.process(rgb_frame)
    
    # Deteksi tangan
    hand_results = hands.process(rgb_frame)
    num_fingers = 0
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            finger_states = [
                1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0
                for tip in finger_tips
            ]
            if sum(finger_states) == 2:
                num_fingers = 2
    
    # Jika dua jari terdeteksi, hentikan pergantian emoji
    if num_fingers == 2:
        change_emoji = False
    else:
        change_emoji = True
    
    # Jika boleh mengganti emoji, pilih emoji baru
    if change_emoji:
        current_emoji = random.choice(emoji_files)
    
    # Gambar emoji di wajah
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            w_box, h_box = int(bboxC.width * w * scale_factor), int(bboxC.height * h * scale_factor)
            x = int(bboxC.xmin * w - (w_box - bboxC.width * w) / 2)
            # y = int(bboxC.ymin * h - (h_box - bboxC.height * h) / 2-40)
            y = int(bboxC.ymin * h - (h_box - bboxC.height * h) / 2 - 45)  # Geser lebih ke atas
            y = max(0, min(y, frame.shape[0] - h_box))  # Pastikan y tetap dalam batas frame


            # Load dan tempel emoji
            emoji = cv2.imread(current_emoji, cv2.IMREAD_UNCHANGED)
            if emoji is not None and emoji.shape[2] == 4:
                emoji = cv2.resize(emoji, (w_box, h_box))
                alpha_s = emoji[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                
                for c in range(3):
                    frame[y:y+h_box, x:x+w_box, c] = (alpha_s * emoji[:, :, c] + alpha_l * frame[y:y+h_box, x:x+w_box, c])
    
    # Tampilkan hasil
    cv2.imshow("Emoji Face Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

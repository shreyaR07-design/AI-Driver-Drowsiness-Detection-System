import cv2
import mediapipe as mp
import math
import pygame
import time
import asyncio
import edge_tts
from playsound import playsound
import os
import threading

# ==========================================
# SOUND SETUP
# ==========================================

pygame.mixer.init()

# Put alarm.mp3 inside project folder
alarm_sound = pygame.mixer.Sound("alarm.mp3")

# ==========================================
# AI VOICE USING EDGE-TTS
# ==========================================

is_speaking = False

def speak_alert():

    global is_speaking

    if is_speaking:
        return

    is_speaking = True

    async def generate_voice():

        communicate = edge_tts.Communicate(
            "Wake up driver",
            voice="en-US-JennyNeural"
        )

        await communicate.save("voice.mp3")

    asyncio.run(generate_voice())

    playsound("voice.mp3")

    if os.path.exists("voice.mp3"):
        os.remove("voice.mp3")

    is_speaking = False

# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# VARIABLES
# ==========================================

EAR_THRESHOLD = 0.20
DROWSY_FRAMES = 15

drowsy_counter = 0
drowsy_events = 0
blink_count = 0

alarm_playing = False
eyes_closed = False

driver_name = "Shreya"

prev_time = time.time()

# ==========================================
# EYE LANDMARKS
# ==========================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==========================================
# FUNCTIONS
# ==========================================

def distance(p1, p2):

    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )

def get_ear(eye_points, landmarks, w, h):

    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    p1 = (int(p1.x * w), int(p1.y * h))
    p2 = (int(p2.x * w), int(p2.y * h))
    p3 = (int(p3.x * w), int(p3.y * h))
    p4 = (int(p4.x * w), int(p4.y * h))
    p5 = (int(p5.x * w), int(p5.y * h))
    p6 = (int(p6.x * w), int(p6.y * h))

    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear, [p1, p2, p3, p4, p5, p6]

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    status = "NO FACE"

    # ==========================================
    # FPS
    # ==========================================

    current_time = time.time()

    fps = int(
        1 / (current_time - prev_time)
    )

    prev_time = current_time

    # ==========================================
    # FACE DETECTION
    # ==========================================

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        landmarks = face_landmarks.landmark

        left_ear, left_points = get_ear(
            LEFT_EYE,
            landmarks,
            w,
            h
        )

        right_ear, right_points = get_ear(
            RIGHT_EYE,
            landmarks,
            w,
            h
        )

        ear = (left_ear + right_ear) / 2

        # ==========================================
        # ORIGINAL GREEN DOT EYES
        # ==========================================

        for eye in [left_points, right_points]:

            for point in eye:

                cv2.circle(
                    frame,
                    point,
                    2,
                    (0, 255, 0),
                    -1
                )

        # ==========================================
        # BLINK COUNT
        # ==========================================

        if ear < EAR_THRESHOLD and not eyes_closed:

            blink_count += 1

            eyes_closed = True

        if ear >= EAR_THRESHOLD:

            eyes_closed = False

        # ==========================================
        # DROWSINESS DETECTION
        # ==========================================

        if ear < EAR_THRESHOLD:

            drowsy_counter += 1

            status = "DROWSY"

            # TRIGGER ONLY ONCE
            if drowsy_counter == DROWSY_FRAMES:

                drowsy_events += 1

                # PLAY SOUND
                if not alarm_playing:

                    alarm_sound.play(-1)

                    alarm_playing = True

                # AI VOICE EVERY TIME
                threading.Thread(
                    target=speak_alert,
                    daemon=True
                ).start()

        else:

            status = "AWAKE"

            drowsy_counter = 0

            # STOP SOUND
            if alarm_playing:

                alarm_sound.stop()

                alarm_playing = False

        # ==========================================
        # STATUS PANEL
        # ==========================================

        cv2.rectangle(
            frame,
            (10, 10),
            (280, 170),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "DRIVER STATUS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Driver : {driver_name}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"EAR : {ear:.2f}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Blinks : {blink_count}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drowsy : {drowsy_events}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

        # ==========================================
        # STATUS TEXT
        # ==========================================

        color = (0, 255, 0)

        if status == "DROWSY":
            color = (0, 0, 255)

        cv2.putText(
            frame,
            f"STATUS : {status}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # ==========================================
        # ALERT BOX
        # ==========================================

        if status == "DROWSY":

            cv2.rectangle(
                frame,
                (100, h - 90),
                (w - 100, h - 30),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (120, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )

    # ==========================================
    # FPS TEXT
    # ==========================================

    cv2.putText(
        frame,
        f"FPS : {fps}",
        (w - 120, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ==========================================
    # QUIT TEXT
    # ==========================================

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ==========================================
    # SHOW WINDOW
    # ==========================================

    cv2.imshow(
        "Driver Drowsiness Detection System",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ==========================================
# END
# ==========================================

alarm_sound.stop()

cap.release()

cv2.destroyAllWindows()
        
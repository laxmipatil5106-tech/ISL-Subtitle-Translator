import os
import json
import time
import threading
import subprocess
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "/Users/laxmi/ISL_Subtitle_Translator/models/isl_model_v2.keras"
LABEL_PATH = "/Users/laxmi/ISL_Subtitle_Translator/models/labels_v2.json"

SEQUENCE_LENGTH = 30
FEATURES = 1662

CONFIDENCE_THRESHOLD = 0.60

SPEECH_COOLDOWN = 2.0


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# STARTUP
# ============================================================

print()
print("==========================================")
print("       ISL V2 REAL-TIME SUBTITLES")
print("==========================================")
print()

print("Loading model...")


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit


if not os.path.exists(LABEL_PATH):

    print("ERROR: Labels not found:")
    print(LABEL_PATH)
    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    labels = json.load(file)


print("Loaded labels:", len(labels))
print()
print("Expected features per frame:", FEATURES)
print("Sequence length:", SEQUENCE_LENGTH)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_keypoints(results):

    # --------------------------------------------------------
    # POSE
    # 33 × 4 = 132
    # --------------------------------------------------------

    if results.pose_landmarks:

        pose = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ]
                for landmark
                in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        pose = np.zeros(
            33 * 4,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # FACE
    # 468 × 3 = 1404
    # --------------------------------------------------------

    if results.face_landmarks:

        face = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
                for landmark
                in results.face_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        face = np.zeros(
            468 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # LEFT HAND
    # 21 × 3 = 63
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        left_hand = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
                for landmark
                in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        left_hand = np.zeros(
            21 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # RIGHT HAND
    # 21 × 3 = 63
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        right_hand = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
                for landmark
                in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        right_hand = np.zeros(
            21 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # COMBINE
    # 132 + 1404 + 63 + 63 = 1662
    # --------------------------------------------------------

    keypoints = np.concatenate(
        [
            pose,
            face,
            left_hand,
            right_hand
        ]
    )

    return keypoints


# ============================================================
# VOICE
# ============================================================

speech_lock = threading.Lock()

last_spoken_word = ""

last_speech_time = 0


def speak_text(text):

    global last_spoken_word
    global last_speech_time

    if text == "":
        return


    current_time = time.time()


    if (
        text == last_spoken_word
        and
        current_time - last_speech_time < SPEECH_COOLDOWN
    ):
        return


    last_spoken_word = text
    last_speech_time = current_time


    def speak():

        with speech_lock:

            try:

                subprocess.run(
                    [
                        "say",
                        text
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            except Exception as error:

                print(
                    "Voice error:",
                    error
                )


    threading.Thread(
        target=speak,
        daemon=True
    ).start()


# ============================================================
# CAMERA
# ============================================================

print("Opening camera...")

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open camera.")

    raise SystemExit


print("Camera opened successfully.")
print()
print("Starting ISL recognition...")
print("Show a sign to the camera.")
print("Press Q to quit.")
print()


# ============================================================
# SEQUENCE
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================================
# DISPLAY VARIABLES
# ============================================================

displayed_word = "Show a sign..."

confidence_value = 0.0


# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================

with mp_holistic.Holistic(

    static_image_mode=False,

    model_complexity=1,

    smooth_landmarks=True,

    refine_face_landmarks=False,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

) as holistic:


    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================

    while cap.isOpened():


        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, frame = cap.read()


        if not success:

            print("Could not read camera frame.")

            continue


        # ----------------------------------------------------
        # MIRROR
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False


        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        results = holistic.process(
            rgb
        )

        rgb.flags.writeable = True


        # ----------------------------------------------------
        # DRAW POSE
        # ----------------------------------------------------

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )


        # ----------------------------------------------------
        # DRAW LEFT HAND
        # ----------------------------------------------------

        if results.left_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )


        # ----------------------------------------------------
        # DRAW RIGHT HAND
        # ----------------------------------------------------

        if results.right_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )


        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        keypoints = extract_keypoints(
            results
        )


        # ----------------------------------------------------
        # FEATURE CHECK
        # ----------------------------------------------------

        if keypoints.shape[0] != FEATURES:

            displayed_word = "FEATURE ERROR"

            cv2.putText(
                frame,
                displayed_word,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "ISL V2",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break

            continue


        # ----------------------------------------------------
        # ADD FRAME
        # ----------------------------------------------------

        sequence.append(
            keypoints
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        if len(sequence) == SEQUENCE_LENGTH:


            input_data = np.expand_dims(
                np.array(
                    sequence,
                    dtype=np.float32
                ),
                axis=0
            )


            prediction = model.predict(
                input_data,
                verbose=0
            )[0]


            class_id = int(
                np.argmax(prediction)
            )


            confidence = float(
                prediction[class_id]
            )


            confidence_value = confidence


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:


                # --------------------------------------------
                # GET PREDICTED WORD
                # --------------------------------------------

                predicted_word = str(
                    labels[class_id]
                )


                # --------------------------------------------
                # UPDATE SUBTITLE DIRECTLY
                # --------------------------------------------

                displayed_word = predicted_word


                # --------------------------------------------
                # TERMINAL
                # --------------------------------------------

                print(
                    f"Prediction: "
                    f"{predicted_word} "
                    f"({confidence * 100:.1f}%)"
                )


                # --------------------------------------------
                # VOICE
                # --------------------------------------------

                speak_text(
                    predicted_word
                )


        # ====================================================
        # USER INTERFACE
        # ====================================================

        height, width = frame.shape[:2]


        # ----------------------------------------------------
        # TOP BAR
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 85),
            (30, 30, 30),
            -1
        )


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "ISL V2",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Confidence: "
            f"{confidence_value * 100:.1f}%",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )


        # ----------------------------------------------------
        # QUIT TEXT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Press Q to quit",
            (
                width - 160,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1
        )


        # ====================================================
        # SUBTITLE AREA
        # ====================================================

        subtitle_height = 125


        cv2.rectangle(
            frame,
            (
                0,
                height - subtitle_height
            ),
            (
                width,
                height
            ),
            (20, 20, 20),
            -1
        )


        # ----------------------------------------------------
        # SUBTITLE
        # ----------------------------------------------------

        subtitle = displayed_word


        if subtitle == "":

            subtitle = "Show a sign..."


        # ----------------------------------------------------
        # TEXT SIZE
        # ----------------------------------------------------

        text_size = cv2.getTextSize(
            subtitle,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            3
        )[0]


        text_x = (
            width - text_size[0]
        ) // 2


        text_y = height - 50


        # ----------------------------------------------------
        # DRAW SUBTITLE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            subtitle,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # SHOW WINDOW
        # ----------------------------------------------------

        cv2.imshow(
            "ISL V2",
            frame
        )


        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("==========================================")
print("           ISL V2 STOPPED")
print("==========================================")
print()

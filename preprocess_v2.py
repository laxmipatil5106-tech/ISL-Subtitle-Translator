import cv2
import mediapipe as mp
import numpy as np
import os
import json
from glob import glob

# ==============================
# SETTINGS
# ==============================

DATASET_DIR = "/Users/laxmi/ISL_Subtitle_Translator/data/isl_dataset"
OUTPUT_DIR = "/Users/laxmi/ISL_Subtitle_Translator/data/processed_v2"

SEQUENCE_LENGTH = 30

# ==============================
# MEDIAPIPE HOLISTIC
# ==============================

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# EXTRACT LANDMARKS
# ==============================

def extract_landmarks(results):

    # LEFT HAND = 21 × 3 = 63
    left_hand = []

    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            left_hand.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])
    else:
        left_hand = [0.0] * 63

    # RIGHT HAND = 21 × 3 = 63
    right_hand = []

    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            right_hand.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])
    else:
        right_hand = [0.0] * 63

    # POSE = 33 × 4 = 132
    pose = []

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            pose.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
    else:
        pose = [0.0] * 132

    # FACE = 468 × 3 = 1404
    face = []

    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            face.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])
    else:
        face = [0.0] * 1404

    # TOTAL = 63 + 63 + 132 + 1404 = 1662
    return np.array(
        left_hand + right_hand + pose + face,
        dtype=np.float32
    )


# ==============================
# PROCESS ONE VIDEO
# ==============================

def process_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open:", video_path)
        return False

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return False

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    sequence = []

    current_frame = 0
    target_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if target_index >= len(frame_indices):
            break

        if current_frame == frame_indices[target_index]:

            image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image.flags.writeable = False

            results = holistic.process(image)

            image.flags.writeable = True

            landmarks = extract_landmarks(results)

            sequence.append(landmarks)

            target_index += 1

        current_frame += 1

    cap.release()

    if len(sequence) != SEQUENCE_LENGTH:
        return False

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (30, 1662):

        print(
            "Unexpected shape:",
            sequence.shape,
            "for",
            video_path
        )

        return False

    np.save(
        output_path,
        sequence
    )

    return True


# ==============================
# MAIN
# ==============================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    label_folders = [
        folder
        for folder in glob(
            os.path.join(DATASET_DIR, "*")
        )
        if os.path.isdir(folder)
    ]

    print()
    print("======================================")
    print("       ISL V2 PREPROCESSING")
    print("======================================")
    print()

    print(
        "Dataset folder:",
        os.path.abspath(DATASET_DIR)
    )

    print(
        "Output folder:",
        os.path.abspath(OUTPUT_DIR)
    )

    print()

    print(
        "Found labels:",
        len(label_folders)
    )

    print()

    total_videos = 0
    processed_videos = 0
    failed_videos = 0

    labels = []

    for label_folder in sorted(label_folders):

        label = os.path.basename(
            label_folder
        )

        labels.append(label)

        output_label_folder = os.path.join(
            OUTPUT_DIR,
            label
        )

        os.makedirs(
            output_label_folder,
            exist_ok=True
        )

        video_files = []

        for extension in [
            "*.mp4",
            "*.MP4",
            "*.avi",
            "*.AVI",
            "*.mov",
            "*.MOV"
        ]:

            video_files.extend(
                glob(
                    os.path.join(
                        label_folder,
                        extension
                    )
                )
            )

        video_files = sorted(video_files)

        print(
            f"Processing '{label}' "
            f"({len(video_files)} videos)"
        )

        total_videos += len(video_files)

        for video_number, video_path in enumerate(
            video_files,
            start=1
        ):

            base_name = os.path.basename(
                video_path
            )

            name_without_extension = os.path.splitext(
                base_name
            )[0]

            filename = (
                name_without_extension + ".npy"
            )

            output_path = os.path.join(
                output_label_folder,
                filename
            )

            if os.path.exists(output_path):

                processed_videos += 1

                print(
                    f"  [{video_number}/{len(video_files)}] "
                    f"Already processed"
                )

                continue

            success = process_video(
                video_path,
                output_path
            )

            if success:

                processed_videos += 1

                print(
                    f"  [{video_number}/{len(video_files)}] "
                    f"Done"
                )

            else:

                failed_videos += 1

                print(
                    f"  [{video_number}/{len(video_files)}] "
                    f"FAILED"
                )

    labels_path = os.path.join(
        OUTPUT_DIR,
        "labels.json"
    )

    with open(
        labels_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            labels,
            file,
            indent=4
        )

    print()
    print("======================================")
    print("      PREPROCESSING COMPLETE")
    print("======================================")
    print()

    print(
        "Total videos :",
        total_videos
    )

    print(
        "Processed    :",
        processed_videos
    )

    print(
        "Failed       :",
        failed_videos
    )

    print(
        "Labels       :",
        len(labels)
    )

    print()
    print(
        "Features per frame : 1662"
    )

    print(
        "Frames per video   : 30"
    )

    print()
    print(
        "V2 processing finished."
    )


if __name__ == "__main__":
    main()
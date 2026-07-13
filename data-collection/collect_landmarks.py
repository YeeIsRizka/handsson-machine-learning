import os
import cv2
import csv
import time
import string
import numpy as np
import mediapipe as mp

# =========================
# Label Configuration
# =========================

one_hand_labels = ['C', 'E', 'I', 'J', 'L', 'O', 'R', 'U', 'V', 'Z']

# Asumsi: label dua tangan adalah huruf A-Z selain label satu tangan.
# Jika ada label yang berbeda, silakan edit list ini.
all_labels = list(string.ascii_uppercase)
two_hand_labels = [label for label in all_labels if label not in one_hand_labels]


# =========================
# Sampling Configuration
# =========================

SAMPLE_INTERVAL = 1.0
PREPARATION_TIME = 5


# =========================
# Initialize MediaPipe Hands
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =========================
# Create CSV Header
# =========================

def create_csv_header():
    header = ["label"]

    for hand in ["left", "right"]:
        for i in range(21):
            header.extend([
                f"{hand}_{i}_x",
                f"{hand}_{i}_y",
                f"{hand}_{i}_z"
            ])

    return header


# =========================
# Save Row to CSV
# =========================

def save_to_csv(filename, data, header):
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(data)


# =========================
# Extract Raw Landmarks
# =========================

def extract_raw_landmarks(results):
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    left_detected = False
    right_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            coords = []

            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            hand_label = handedness.classification[0].label

            if hand_label == "Left":
                left_hand = coords
                left_detected = True

            elif hand_label == "Right":
                right_hand = coords
                right_detected = True

    return left_hand, right_hand, left_detected, right_detected


# =========================
# Reshape Landmark
# =========================

def reshape_landmarks(flat_landmarks):
    return np.array([
        [
            flat_landmarks[i],
            flat_landmarks[i + 1],
            flat_landmarks[i + 2]
        ]
        for i in range(0, len(flat_landmarks), 3)
    ], dtype=np.float32)


# =========================
# Translation Normalization
# =========================

def translation_normalization(flat_landmarks):
    coords = reshape_landmarks(flat_landmarks)

    wrist = coords[0]
    translated_coords = coords - wrist

    return translated_coords.flatten().tolist()


# =========================
# Scale Normalization
# =========================

def scale_normalization(flat_translation_landmarks):
    coords = reshape_landmarks(flat_translation_landmarks)

    v_max = coords.max(axis=0)
    v_min = coords.min(axis=0)

    scale_factor = np.linalg.norm(v_max - v_min, ord=2)

    if scale_factor == 0:
        scale_factor = 1e-6

    scaled_coords = coords / scale_factor

    return scaled_coords.flatten().tolist()


# =========================
# Count Detected Hands
# =========================

def count_detected_hands(left_detected, right_detected):
    return int(left_detected) + int(right_detected)


# =========================
# Progress Bar
# =========================

def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    empty = width - filled
    bar = "█" * filled + "-" * empty
    return f"[{bar}] {current}/{total}"


# =========================
# Detect Mode from Label
# =========================

def detect_mode_from_label(label):
    label = label.upper()

    if label in one_hand_labels:
        return 1

    if label in two_hand_labels:
        return 2

    return None


# =========================
# Input Label
# =========================

def get_label_and_mode():
    print("\nDaftar label satu tangan:")
    print("  " + ", ".join(one_hand_labels))

    print("\nDaftar label dua tangan:")
    print("  " + ", ".join(two_hand_labels))

    while True:
        label = input("\nMasukkan label gesture yang ingin diambil: ").strip().upper()

        if label == "":
            print("Label tidak boleh kosong.")
            continue

        mode = detect_mode_from_label(label)

        if mode is None:
            print("Label tidak dikenali.")
            print("Masukkan label yang ada pada daftar satu tangan atau dua tangan.")
            continue

        return label, mode


# =========================
# Input Number of Samples
# =========================

def get_num_samples():
    while True:
        try:
            num_samples = int(input("Masukkan jumlah sampel yang ingin diambil: "))

            if num_samples > 0:
                return num_samples

            print("Jumlah sampel harus lebih dari 0.")

        except ValueError:
            print("Masukkan angka yang valid.")


# =========================
# Draw Frame Information
# =========================

def draw_frame_info(frame, label, mode, sample_count, num_samples, preparing=False, countdown=None):
    mode_text = "One Hand" if mode == 1 else "Two Hands"

    cv2.putText(
        frame,
        f"Label: {label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Mode: {mode_text}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sample: {sample_count}/{num_samples}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    if preparing and countdown is not None:
        cv2.putText(
            frame,
            f"Bersiap... {countdown}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


# =========================
# Print Header
# =========================

def print_header():
    print("\n" + "=" * 70)
    print(" HAND LANDMARK DATA COLLECTION")
    print(" Mode otomatis berdasarkan label")
    print(" Scale Normalization Only")
    print("=" * 70)


# =========================
# Collect Samples
# =========================

def collect_samples(label, mode, num_samples):
    output_file = "bisindo_landmarks_scaled.csv"
    header = create_csv_header()

    print("\nKonfigurasi:")
    print(f"  Label        : {label}")
    print(f"  Mode         : {'Satu tangan' if mode == 1 else 'Dua tangan'}")
    print(f"  Target sampel: {num_samples}")
    print("\nOutput file:")
    print(f"  Scale Normalized      : {output_file}")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\nERROR: Kamera tidak dapat dibuka.")
        return

    print("\nKontrol:")
    print("  Tekan 's' pada window kamera untuk mulai")
    print("  Tekan 'q' pada window kamera untuk keluar")
    print("-" * 70)

    collecting = False
    preparing = False
    sample_count = 0
    start_time = None
    prep_start_time = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Gagal membaca frame dari kamera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        left_raw, right_raw, left_detected, right_detected = extract_raw_landmarks(results)
        detected_count = count_detected_hands(left_detected, right_detected)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        countdown = None

        if preparing:
            elapsed_prep = time.time() - prep_start_time
            countdown = max(0, int(PREPARATION_TIME - elapsed_prep))

            if elapsed_prep >= PREPARATION_TIME:
                preparing = False
                collecting = True
                sample_count = 0
                start_time = time.time()

                print("\nPengambilan data dimulai.")
                print("Pastikan posisi tangan sesuai label.")
                print("-" * 70)

        draw_frame_info(
            frame=frame,
            label=label,
            mode=mode,
            sample_count=sample_count,
            num_samples=num_samples,
            preparing=preparing,
            countdown=countdown
        )

        cv2.imshow("Hand Landmark Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nKeluar dari program.")
            break

        elif key == ord("s") and not collecting and not preparing:
            print("\nPersiapkan posisi tangan. Mulai dalam 5 detik.")
            preparing = True
            prep_start_time = time.time()

        if collecting and sample_count < num_samples:
            current_time = time.time()

            if current_time - start_time >= SAMPLE_INTERVAL:
                if mode == 1:
                    valid_sample = detected_count == 1
                else:
                    valid_sample = detected_count == 2

                if valid_sample:
                    if mode == 1:
                        # Format tetap 126 fitur.
                        # Tangan yang tidak digunakan diisi 0.
                        if left_detected:
                            right_raw = [0.0] * 63
                        elif right_detected:
                            left_raw = [0.0] * 63

                    # Translation
                    if left_detected:
                        left_translation = translation_normalization(left_raw)
                    else:
                        left_translation = [0.0] * 63

                    if right_detected:
                        right_translation = translation_normalization(right_raw)
                    else:
                        right_translation = [0.0] * 63

                    # Scale row
                    if left_detected:
                        left_scale = scale_normalization(left_translation)
                    else:
                        left_scale = [0.0] * 63

                    if right_detected:
                        right_scale = scale_normalization(right_translation)
                    else:
                        right_scale = [0.0] * 63

                    scale_row = [label] + left_scale + right_scale

                    save_to_csv(output_file, scale_row, header)

                    sample_count += 1

                    print(
                        f"{progress_bar(sample_count, num_samples)} "
                        f"Label '{label}' disimpan."
                    )

                else:
                    if mode == 1:
                        print(
                            f"Skip: terdeteksi {detected_count} tangan. "
                            "Label ini membutuhkan tepat 1 tangan."
                        )
                    else:
                        print(
                            f"Skip: terdeteksi {detected_count} tangan. "
                            "Label ini membutuhkan tepat 2 tangan."
                        )

                start_time = current_time

        if collecting and sample_count >= num_samples:
            print("\n" + "=" * 70)
            print("Pengambilan data selesai.")
            print(f"Label        : {label}")
            print(f"Mode         : {'Satu tangan' if mode == 1 else 'Dua tangan'}")
            print(f"Total sampel : {sample_count}/{num_samples}")
            print(f"Scale Normalized      : {output_file}")
            print("=" * 70)
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# Main Program
# =========================

def main():
    print_header()

    label, mode = get_label_and_mode()
    num_samples = get_num_samples()

    collect_samples(label, mode, num_samples)


# =========================
# Run Program
# =========================

if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIG
# =========================
MODEL_PATH = "gesture_recognizer.task"   # حط الملف هنا نفس المسار
CAMERA_INDEX = 0                         # 0 غالباً هي webcam
NUM_HANDS = 1                            # عدد اليدين
CONF_THRESHOLD = 0.50                    # أقل score باش نعتبره gesture صحيح

# =========================
# MAIN
# =========================
def main():
    # MediaPipe options
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ ما قدرش يفتح الكاميرا. بدّل CAMERA_INDEX ولا شوف permissions.")
        return

    ts_ms = 0  # timestamp بالميلي ثانية

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ ما قدرش يقرا frame من الكاميرا.")
                break

            # تحويل BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            # تحديث timestamp (تقريباً 30fps)
            ts_ms += 33

            # Recognition
            result = recognizer.recognize_for_video(mp_image, ts_ms)

            label = ""
            score = 0.0

            # خذ أفضل gesture
            if result.gestures and len(result.gestures) > 0 and len(result.gestures[0]) > 0:
                best = result.gestures[0][0]
                label = best.category_name
                score = best.score

            # عرض text فوق الفيديو
            display_text = f"{label} ({score:.2f})" if label else "No gesture"
            cv2.putText(
                frame, display_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

            # Commands (غير إذا score كافي)
            if label and score >= CONF_THRESHOLD:
                if label == "Open_Palm":
                    print("A")
                elif label == "Thumb_Up":
                    print("B")
                elif label == "Closed_Fist":
                    print("C")

            cv2.imshow("Gesture Recognizer", frame)

            # Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

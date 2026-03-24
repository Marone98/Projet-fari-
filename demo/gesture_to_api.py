import time
import base64
import cv2
import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ======================
# CONFIGURATION
# ======================
API_BASE = "http://127.0.0.1:5000"
DRAW_GRID_URL = f"{API_BASE}/draw_grid"
PLAY_URL = f"{API_BASE}/play"

MODEL_PATH = "gesture_recognizer.task"   # Doit être dans le même dossier que ce script

# Paramètres de la grille (modifiables)
GRID_CENTER = [0.15, 0.15]
GRID_SIZE = [0.10]

# Anti-spam : évite d’envoyer trop de requêtes
COOLDOWN_SECONDS = 1.0

# Mapping gestes → commandes
# Exemple : Open_Palm = A, Thumb_Up = B, Closed_Fist = C
GESTURE_TO_CMD = {
    "Open_Palm": "A",      # Exemple : dessiner la grille
    "Thumb_Up": "B",       # Exemple : jouer un coup
    "Closed_Fist": "C",    # Exemple : reset / stop (pour l’instant juste un print)
}

last_sent = {}  # commande -> dernier temps d’envoi


def can_send(cmd: str) -> bool:
    """Vérifie si on peut envoyer la commande (cooldown)."""
    now = time.time()
    t = last_sent.get(cmd, 0)
    if now - t >= COOLDOWN_SECONDS:
        last_sent[cmd] = now
        return True
    return False


def post_draw_grid():
    """Appelle l’API pour dessiner la grille."""
    payload = {"center": GRID_CENTER, "size": GRID_SIZE}
    r = requests.post(DRAW_GRID_URL, json=payload, timeout=5)
    print("➡️ draw_grid :", r.status_code, r.text)


def post_play(frame_bgr):
    """Envoie une image à l’API /play (encodée en base64)."""
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        print("Échec de l’encodage de l’image")
        return

    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    payload = {"image": "data:image/png;base64," + img_b64}

    r = requests.post(PLAY_URL, json=payload, timeout=10)
    print("➡️ play :", r.status_code)
    print(r.text)


def main():
    # Initialisation du Gesture Recognizer MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d’ouvrir la caméra. Essayez VideoCapture(1) ou fermez Teams/Zoom.")
        return

    ts = 0
    print("Démarrage : gestes → API")
    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Impossible de lire une frame depuis la caméra.")
            break

        # MediaPipe attend des images RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        ts += 33  # millisecondes
        result = recognizer.recognize_for_video(mp_image, ts)

        label = None
        score = 0.0
        if result.gestures and len(result.gestures[0]) > 0:
            g = result.gestures[0][0]
            label = g.category_name
            score = float(g.score)

        # Affichage du geste détecté
        if label:
            cv2.putText(
                frame,
                f"{label} ({score:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cmd = GESTURE_TO_CMD.get(label)
            if cmd and score >= 0.55:  # seuil de confiance
                if can_send(cmd):
                    print(f"🟢 Geste détecté = {label} → Commande = {cmd}")

                    # Liaison geste → API
                    if cmd == "A":
                        post_draw_grid()
                    elif cmd == "B":
                        post_play(frame)
                    elif cmd == "C":
                        print("🛑 Commande C (actuellement sans action)")

        cv2.imshow("Gesture -> API", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()

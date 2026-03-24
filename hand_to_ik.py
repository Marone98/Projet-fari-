"""
Contrôle robot (simulation) avec MediaPipe Hands + IK (PyRoki) + Viser.
- 👍 Thumb Up  : ENABLE (le robot suit la position du bout de l'index)
- ✊ Fist      : FREEZE (arrêt du suivi)
- 👋 Open hand : CENTER (retour au centre)
Appuie sur 'q' pour quitter.
"""

import time
import numpy as np
import cv2
import mediapipe as mp

import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import pyroki_snippets as pks


# =========================
# CONFIG
# =========================
TARGET_LINK_NAME = "panda_hand"

BASE_POS = np.array([0.55, 0.00, 0.35])   # centre (viser)
SCALE_X = 0.35                            # amplitude X
SCALE_Y = 0.35                            # amplitude Y
Z_FIXED = 0.35                            # Z fixe (stable)
SMOOTH_ALPHA = 0.25                       # lissage (0..1)


def smooth(prev: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return (1 - alpha) * prev + alpha * new


# =========================
# Gestures (sans modèle .task)
# =========================
def is_fist(lm) -> bool:
    # doigts repliés: tip plus bas que pip (y plus grand => plus bas à l'écran)
    return (lm[8].y > lm[6].y and lm[12].y > lm[10].y and lm[16].y > lm[14].y and lm[20].y > lm[18].y)


def is_open_hand(lm) -> bool:
    # doigts tendus
    return (lm[8].y < lm[6].y and lm[12].y < lm[10].y and lm[16].y < lm[14].y and lm[20].y < lm[18].y)


def is_thumb_up(lm) -> bool:
    # pouce "haut" et index/majeur repliés (simple + robuste)
    thumb_up = lm[4].y < lm[3].y
    index_down = lm[8].y > lm[6].y
    middle_down = lm[12].y > lm[10].y
    return thumb_up and index_down and middle_down


def main():
    # --- Robot + Viser
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(BASE_POS),
        wxyz=(1, 0, 0, 0),
    )

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # --- Camera + MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la caméra. Essaie VideoCapture(1) ou ferme Teams/Zoom.")
        return

    enabled = False
    filtered_pos = BASE_POS.copy()

    print("=== CONTROLES ===")
    print("👍 Thumb Up  : ENABLE (suivi index)")
    print("✊ Fist      : FREEZE")
    print("👋 Open hand : CENTER")
    print("q : quitter")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        status_txt = "NONE"

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark

            # 1) Reconnaissance gestes
            if is_thumb_up(lm):
                enabled = True
                status_txt = "ENABLE 👍"
            elif is_fist(lm):
                enabled = False
                status_txt = "FREEZE ✊"
            elif is_open_hand(lm):
                filtered_pos = BASE_POS.copy()
                ik_target.position = tuple(filtered_pos)
                status_txt = "CENTER 👋"

            # 2) Si ENABLE => suivre le bout de l'index
            if enabled:
                x = lm[8].x  # 0..1
                y = lm[8].y  # 0..1

                X = BASE_POS[0] + (x - 0.5) * SCALE_X
                Y = BASE_POS[1] + (0.5 - y) * SCALE_Y
                Z = Z_FIXED

                target_pos = np.array([X, Y, Z])
                filtered_pos = smooth(filtered_pos, target_pos, SMOOTH_ALPHA)
                ik_target.position = tuple(filtered_pos)

        # --- IK solve
        start = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=TARGET_LINK_NAME,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )
        elapsed = (time.time() - start) * 1000
        timing_handle.value = 0.9 * timing_handle.value + 0.1 * elapsed

        urdf_vis.update_cfg(solution)

        # --- Affichage
        cv2.putText(frame, status_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand -> IK (ThumbUp/ Fist/ OpenHand)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
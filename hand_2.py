"""
6 Gestures -> Robot Directions (Simulation)
RIGHT   = X+
LEFT    = X-
FORWARD = Y+
BACK    = Y-
UP      = Z+
DOWN    = Z-
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

BASE_POS = np.array([0.55, 0.00, 0.35])
STEP = 0.03
SMOOTH = 0.25
COOLDOWN = 0.25

# Safety limits
X_MIN, X_MAX = 0.35, 0.75
Y_MIN, Y_MAX = -0.35, 0.35
Z_MIN, Z_MAX = 0.20, 0.60


def clamp(p):
    return np.array([
        np.clip(p[0], X_MIN, X_MAX),
        np.clip(p[1], Y_MIN, Y_MAX),
        np.clip(p[2], Z_MIN, Z_MAX),
    ])


def smooth(prev, new):
    return (1 - SMOOTH) * prev + SMOOTH * new


# =========================
# Gesture Rules
# =========================
def is_fist(lm):
    return (lm[8].y > lm[6].y and lm[12].y > lm[10].y and
            lm[16].y > lm[14].y and lm[20].y > lm[18].y)


def is_open(lm):
    return (lm[8].y < lm[6].y and lm[12].y < lm[10].y and
            lm[16].y < lm[14].y and lm[20].y < lm[18].y)


def is_peace(lm):  # UP (Z+)
    return (lm[8].y < lm[6].y and lm[12].y < lm[10].y and
            lm[16].y > lm[14].y and lm[20].y > lm[18].y)


def is_down(lm):  # DOWN (Z-)
    return (lm[8].y - lm[0].y) > 0.25


def direction_lr(lm):
    dx = lm[8].x - lm[0].x
    if dx > 0.10:
        return "RIGHT"
    if dx < -0.10:
        return "LEFT"
    return None


def direction_fb(lm):
    dy = lm[8].y - lm[0].y
    if dy < -0.10:
        return "FORWARD"
    if dy > 0.10:
        return "BACK"
    return None


# =========================
# MAIN
# =========================
def main():

    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=tuple(BASE_POS), wxyz=(1, 0, 0, 0)
    )

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(False, 1, 1, 0.6, 0.6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    target = BASE_POS.copy()
    filtered = BASE_POS.copy()
    last_cmd = 0

    print("RIGHT X+ | LEFT X- | FORWARD Y+ | BACK Y- | UP Z+ | DOWN Z-")

    while True:

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        status = "NONE"

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            now = time.time()

            if now - last_cmd > COOLDOWN:

                # RIGHT / LEFT (X)
                lr = direction_lr(lm)
                if lr == "RIGHT":
                    target += np.array([STEP, 0, 0])
                    status = "RIGHT X+"
                    last_cmd = now

                elif lr == "LEFT":
                    target += np.array([-STEP, 0, 0])
                    status = "LEFT X-"
                    last_cmd = now

                # FORWARD / BACK (Y)
                fb = direction_fb(lm)
                if fb == "FORWARD":
                    target += np.array([0, STEP, 0])
                    status = "FORWARD Y+"
                    last_cmd = now

                elif fb == "BACK":
                    target += np.array([0, -STEP, 0])
                    status = "BACK Y-"
                    last_cmd = now

                # UP / DOWN (Z)
                if is_peace(lm):
                    target += np.array([0, 0, STEP])
                    status = "UP Z+"
                    last_cmd = now

                elif is_down(lm):
                    target += np.array([0, 0, -STEP])
                    status = "DOWN Z-"
                    last_cmd = now

            target = clamp(target)

        filtered = smooth(filtered, target)
        ik_target.position = tuple(filtered)

        solution = pks.solve_ik(
            robot=robot,
            target_link_name=TARGET_LINK_NAME,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        urdf_vis.update_cfg(solution)

        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("6 Directions Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
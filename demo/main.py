# main.py

import argparse
import signal
import atexit
from flask import Flask, request, jsonify
import base64
import numpy as np
from vision import image_to_tictactoe_grid
from tictactoe_engine import find_best_move
from PIL import Image
import io
import cv2
import os
import asyncio
import sys
from flask import render_template

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



def joint_to_SE3(origin, x_point, y_point):
    # Compute the Cartesian coordinates of the origin and the two points
    # Define basis vectors for the new frame
    x_prime = x_point - origin
    x_prime = x_prime / np.linalg.norm(x_prime)  # Normalize to unit vector
    
    # Vector in the plane but orthogonal to x_prime
    temp_vec = y_point - origin
    z_prime = np.cross(x_prime, temp_vec)
    z_prime = z_prime / np.linalg.norm(z_prime)  # Normalize to unit vector
    
    # y_prime is orthogonal to both x_prime and z_prime
    y_prime = np.cross(z_prime, x_prime)
    y_prime = y_prime / np.linalg.norm(y_prime)  # Normalize to unit vector
    
    # Rotation matrix
    R = np.column_stack((x_prime, y_prime, z_prime))
    
    # Translation vector
    t = origin
    
    # Create SE(3) transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return sm.SE3(T)

app = Flask(__name__)

def initialize_app(modes, robot_ip=None):
    print("simulation api only")
    return app

SIMULATION_ONLY =True
@app.route('/draw_grid', methods=['POST'])
def draw_grid():
    data = request.get_json()
    center = data.get('center')
    size = data.get('size')

    if not center or not size:
        return jsonify({"message": "Invalid input"}), 400

    if SIMULATION_ONLY:
        print(f"[SIMULATION] draw_grid center={center}, size={size}")
        return jsonify({"message": "Grid generated successfully (simulation)"}), 200

    # REAL robot
    center_position = sm.SE3(center[0], center[1], 0)
    size_value = size[0]
    oxoplayer.draw_grid(center_position, size_value)

    return jsonify({"message": "Grid generated successfully"}), 200


@app.route('/play', methods=['POST'])
def play():
    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({"message": "Invalid input"}), 400

    if SIMULATION_ONLY:
        print("[SIMULATION] play called")
        return jsonify({
            "grid": [],
            "move": None,
            "status": "simulation",
            "winner": None
        }), 200

    # REAL robot logic
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)

    response = oxoplayer.play(image)
    return jsonify(response), 200

@app.route('/')
def home():
    return render_template("index.html")

    


def on_exit():
    print("Terminal closed. Performing cleanup...")
    # Call the specific function you need
    if oxoplayer is not None:
        try:
            oxoplayer.cleanup()
        except Exception as e:
            print("Cleanup ignored:", e)

  # Assuming you have a cleanup method in OXOPlayer
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Tic-Tac-Toe Flask app.")
    parser.add_argument('--modes', type=str, nargs='+', required=True,
                        choices=["SIMULATION", "REAL"],
                        help="Modes to run the application in.")
    parser.add_argument('--robot_ip', type=str, help="IP address of the robot for REAL mode.")
    args = parser.parse_args()

    # init app first
    initialize_app(args.modes, args.robot_ip)

    # Register the exit handler BEFORE starting server
    signal.signal(signal.SIGINT, lambda sig, frame: on_exit())
    signal.signal(signal.SIGTERM, lambda sig, frame: on_exit())
    atexit.register(on_exit)

    # Start server ONCE
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


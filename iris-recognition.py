import cv2
import mediapipe as mp
import os
import time
import webbrowser
import numpy as np


save_path = "IrisCaptures"
os.makedirs(save_path, exist_ok=True)
print(" Images will be saved in:", os.path.abspath(save_path))


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Could not access laptop camera.")
    exit()


mp_face_mesh = mp.solutions.face_mesh
RIGHT_IRIS = [474, 475, 476, 477]


def get_iris_color(cropped_eye):
    if cropped_eye is None or cropped_eye.size == 0:
        return "Unknown"


    eye_resized = cv2.resize(cropped_eye, (50, 50))
    avg_color = np.mean(eye_resized.reshape(-1, 3), axis=0)  # [B, G, R]

    b, g, r = avg_color

    hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # Heuristic color classification
    if v < 50:
        return "Black"
    elif s < 40 and v > 150:
        return "Gray"
    elif h < 25 and v > 50:
        return "Brown"
    elif 25 < h < 90:
        return "Green"
    elif h >= 90:
        return "Blue"
    else:
        return "Unknown"


with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Frame not captured. Check your camera connection.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        right_eye_crop = None
        iris_color = "Unknown"

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                h, w, _ = frame.shape

                # Collect right iris landmark points
                iris_points = np.array([
                    [int(face_landmarks.landmark[i].x * w),
                     int(face_landmarks.landmark[i].y * h)]
                    for i in RIGHT_IRIS
                ])

                # Draw right iris landmarks
                for (x, y) in iris_points:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Compute bounding box for the iris
                x_min, y_min = np.min(iris_points, axis=0)
                x_max, y_max = np.max(iris_points, axis=0)

                # Add padding around the iris
                pad = 15
                x_min = max(x_min - pad, 0)
                y_min = max(y_min - pad, 0)
                x_max = min(x_max + pad, w)
                y_max = min(y_max + pad, h)

                # Crop the right eye region
                right_eye_crop = frame[y_min:y_max, x_min:x_max]

                # Guess iris color
                iris_color = get_iris_color(right_eye_crop)

                # Display the color guess on screen
                cv2.putText(frame, f"Iris: {iris_color}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Right Eye Detection + Iris Color", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        # Press 'c' to capture image and name it
        if key == ord('c'):
            if right_eye_crop is not None:
                name = input(" Enter name for this eye capture: ").strip()
                if name == "":
                    name = f"eye_{int(time.time())}"

                filename = os.path.join(save_path, f"{name}.jpg")
                cv2.imwrite(filename, right_eye_crop)
                print(f" Saved cropped right eye as {filename}")
                print(f"Detected Iris Color: {iris_color}")

                webbrowser.open(os.path.abspath(save_path))
            else:
                print(" No iris detected! Try again.")

        # Press ESC to exit
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()

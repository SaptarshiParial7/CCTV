import cv2
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import Button
from threading import Thread
import hashlib

# Absolute path for detected faces and log file
base_dir = os.path.abspath("cctv_faces")
os.makedirs(base_dir, exist_ok=True)

log_file = os.path.join(base_dir, "cctv_log.csv")

# Create a CSV file if it doesn't exist
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Face_ID", "Image_Name", "Date", "Time"]).to_csv(log_file, index=False)

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize global variables
cap = None
running = False
face_detected = False

def compute_face_hash(face_image):
    """Compute a hash for the given face image."""
    resized_face = cv2.resize(face_image, (50, 50))  # Resize for consistency
    face_hash = hashlib.md5(resized_face.tobytes()).hexdigest()  # Hash the image bytes
    return face_hash

def is_face_already_logged(face_hash, date):
    """Check if the face is already logged for the given day."""
    try:
        df = pd.read_csv(log_file)
        existing_logs = df[(df["Face_ID"] == face_hash) & (df["Date"] == date)]
        return not existing_logs.empty
    except pd.errors.EmptyDataError:
        # Ignore empty CSV files
        return False
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return False

def log_face(face_hash, img_name, date, time):
    """Log a new face in the CSV file."""
    try:
        df = pd.read_csv(log_file)
        new_entry = pd.DataFrame([{
            "Face_ID": face_hash,
            "Image_Name": img_name,
            "Date": date,
            "Time": time
        }])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(log_file, index=False)
    except pd.errors.EmptyDataError:
        # Ignore empty CSV files
        pass
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def start_camera():
    """Start the camera and face detection."""
    global cap, running, face_detected

    if running:
        return  # Avoid multiple threads

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    running = True

    def detect_faces():
        global face_detected
        print("Camera started. Press 'q' in the console to quit.")

        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                # Extract the face
                face = frame[y:y+h, x:x+w]

                # Compute a hash for the face
                face_hash = compute_face_hash(face)

                # Get the current date and time
                timestamp = datetime.now()
                date = timestamp.strftime("%Y-%m-%d")
                time = timestamp.strftime("%H:%M:%S")

                # Check if the face is already logged today
                if not face_detected and not is_face_already_logged(face_hash, date):
                    # Save the face image
                    img_name = f"face_{date}_{time.replace(':', '')}.jpg"
                    img_path = os.path.join(base_dir, img_name)
                    cv2.imwrite(img_path, face)

                    # Log the face
                    log_face(face_hash, img_name, date, time)

                    # Mark the face as detected
                    face_detected = True

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("CCTV - Face Detection", frame)

            # Quit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        stop_camera()

    # Run the detection in a separate thread
    Thread(target=detect_faces, daemon=True).start()

def stop_camera():
    """Stop the camera and close all windows."""
    global cap, running, face_detected

    running = False
    face_detected = False
    if cap:
        cap.release()
        cap = None
    cv2.destroyAllWindows()
    print("Camera stopped. Data logged in", log_file)

# Create the GUI
app = tk.Tk()
app.title("CCTV Face Detection")

start_button = Button(app, text="Start Camera", command=start_camera, width=20)
start_button.pack(pady=10)

stop_button = Button(app, text="Stop Camera", command=stop_camera, width=20)
stop_button.pack(pady=10)

app.mainloop()


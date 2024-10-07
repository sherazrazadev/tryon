import cv2
import mediapipe as mp
import cvzone
import numpy as np
import os
import requests
from flask import Flask, render_template, request, jsonify, Response
import logging
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up logging to only show errors
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Suppress Mediapipe logs specifically
logging.getLogger('absl').setLevel(logging.ERROR)

app = Flask(__name__)

# Ensure the necessary directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/output', exist_ok=True)

# Initialize Mediapipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Global variables to hold the processed PNG path and overlay
glasses_png_path = None
overlay = None

# Resize the input image for consistent detection (if image is too large)
def resize_image_for_detection(image, max_width=800):
    h, w = image.shape[:2]
    if w > max_width:
        scale_ratio = max_width / w
        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

# Function to detect facial landmarks using mediapipe
def get_landmarks(frame):
    try:
        # Convert the image to RGB as Mediapipe expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using the face mesh model
        results = face_mesh.process(rgb_frame)

        # Extract landmarks
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Convert normalized landmarks to pixel coordinates
            h, w, _ = frame.shape
            landmarks_pixel = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            return landmarks_pixel
        else:
            return []
    except Exception as e:
        print(f"Error detecting landmarks: {e}")
        return []

# Function to remove the background from an image using remove.bg API
def remove_background(image_path, api_key, output_path):
    try:
        url = "https://api.remove.bg/v1.0/removebg"
        with open(image_path, 'rb') as image_file:
            response = requests.post(
                url,
                files={'image_file': image_file},
                data={'size': 'auto'},
                headers={'X-Api-Key': api_key},
            )
        if response.status_code == requests.codes.ok:
            # Save the background-removed image
            with open(output_path, 'wb') as out_file:
                out_file.write(response.content)

            # Load the image with transparent background
            img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
            return output_path
        else:
            print(f"Error with remove.bg API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error removing background: {e}")
        return None

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global glasses_png_path, overlay
        try:
            # Handle image upload and background removal here
            file = request.files['glasses_image']
            file_ext = file.filename.split('.')[-1].lower()
            if file_ext not in ['jpg', 'jpeg', 'png']:
                return jsonify({'status': 'error', 'message': 'Please upload a valid image format (jpg, jpeg, png)'})
            
            # Save the uploaded image
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            # Call remove.bg API to remove the background
            api_key = 'MosFyM4XVWkjN9p1fLLZX5iY'  # Replace with your actual remove.bg API key
            output_path = os.path.join('static/output', 'output.png')
            glasses_png_path = remove_background(file_path, api_key, output_path)

            if glasses_png_path:
                # Load the processed image as overlay
                overlay = cv2.imread(glasses_png_path, cv2.IMREAD_UNCHANGED)

                return jsonify({'status': 'success', 'image_url': '/' + glasses_png_path})
            else:
                return jsonify({'status': 'error', 'message': 'Background removal failed'})
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'status': 'error', 'message': 'An error occurred during processing.'})

    return render_template('index2.html')

# glasses_png_path = 'static/output/output.png'

# overlay = cv2.imread(glasses_png_path, cv2.IMREAD_UNCHANGED)

# Route for accessing the webcam and applying the overlay
@app.route('/webcam_feed')
def webcam_feed():
    def generate_frames():
        global overlay
        try:
            cap = cv2.VideoCapture(0)  # Open the webcam

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to a manageable size for face detection
                resized_frame = resize_image_for_detection(frame)

                # Get facial landmarks
                landmarks = get_landmarks(resized_frame)

                if overlay is not None and landmarks:
                    # Extract the left and right eye landmarks (Mediapipe face mesh landmark indices)
                    left_eye = [landmarks[i] for i in range(33, 42)]  # Left eye
                    right_eye = [landmarks[i] for i in range(362, 372)]  # Right eye

                    # Extract the nose bridge landmark (landmark 1 or 168 for nose bridge in Mediapipe)
                    nose_bridge = landmarks[1]  # Use the nose bridge for dynamic y positioning

                    if len(left_eye) > 0 and len(right_eye) > 0:
                        # Compute the center of the left and right eye
                        left_eye_center = np.mean(left_eye, axis=0).astype(int)
                        right_eye_center = np.mean(right_eye, axis=0).astype(int)

                        # Calculate the distance between the eyes
                        eye_width = np.linalg.norm(right_eye_center - left_eye_center).astype(int)

                        # Resize the glasses proportionally based on the distance between the eyes
                        original_glasses_width = overlay.shape[1]
                        original_glasses_height = overlay.shape[0]

                        # Scale the glasses width to match the eye width with a larger scaling factor
                        glasses_width = int(eye_width * 2.6)  # Adjust this factor for appropriate size

                        # Maintain aspect ratio while resizing
                        aspect_ratio = original_glasses_height / original_glasses_width
                        glasses_height = int(glasses_width * aspect_ratio)

                        # Calculate the midpoint between the two eyes to position the glasses
                        eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)

                        # Use the nose bridge landmark for vertical positioning (align glasses above nose)
                        glasses_x = eye_center[0] - int(glasses_width / 2)
                        
                        # Dynamically adjust glasses height based on the nose bridge location
                        glasses_y = nose_bridge[1] - int(glasses_height / 2) - 40  # Adjust by 20 pixels upwards

                        # Resize the overlay (glasses) proportionally
                        overlay_resize = cv2.resize(overlay, (glasses_width, glasses_height))

                        # Overlay the resized glasses image onto the frame
                        frame = cvzone.overlayPNG(resized_frame, overlay_resize, [glasses_x, glasses_y])

                # Encode the frame as PNG
                ret, buffer = cv2.imencode('.png', frame)
                frame = buffer.tobytes()

                # Yield the frame in a streaming response
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

            cap.release()
        except Exception as e:
            print(f"Error during webcam feed: {e}")

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Flask app starting on http://127.0.0.1:5000 or http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)



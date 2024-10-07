from flask import Flask, render_template, request, jsonify, Response
import cv2
import cvzone
import requests
import os

app = Flask(__name__)

# Global variables to hold the processed PNG path and overlay
glasses_png_path = None
overlay = None

# Function to remove the background from an image
def remove_background(image_path, api_key, output_path):
    url = "https://api.remove.bg/v1.0/removebg"
    with open(image_path, 'rb') as image_file:
        response = requests.post(
            url,
            files={'image_file': image_file},
            data={'size': 'auto'},
            headers={'X-Api-Key': api_key},
        )
    if response.status_code == requests.codes.ok:
        with open(output_path, 'wb') as out_file:
            out_file.write(response.content)
        return output_path
    else:
        return None

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global glasses_png_path, overlay
        # Handle image upload and background removal here
        file = request.files['glasses_image']
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['jpg', 'jpeg', 'png']:
            return jsonify({'status': 'error', 'message': 'Please upload a valid image format (jpg, jpeg, png)'})
        
        # Save the uploaded image
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)

        # Call remove.bg API to remove the background , MosFyM4XVWkjN9p1fLLZX5iY
        api_key = 'Q1zzpzfqEq2oxpc7pBkmXJRy'  # Replace with your actual remove.bg API key
        output_path = f'static/output/{os.path.splitext(file.filename)[0]}.png'
        glasses_png_path = remove_background(file_path, api_key, output_path)

        if glasses_png_path:
            # Load the processed image as overlay
            overlay = cv2.imread(glasses_png_path, cv2.IMREAD_UNCHANGED)
            return jsonify({'status': 'success', 'image_url': '/' + glasses_png_path})
        else:
            return jsonify({'status': 'error', 'message': 'Background removal failed'})

    return render_template('index.html')

# Route for accessing the webcam and applying the overlay
@app.route('/webcam_feed')
def webcam_feed():
    def generate_frames():
        global overlay
        # Open the webcam
        cap = cv2.VideoCapture(0)
        # Load the face detection model
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            # Read the current frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale for face detection
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5)

            # For each detected face, apply the overlay
            if overlay is not None:
                for (x, y, w, h) in faces:
                    # Resize the overlay to fit the face width
                    overlay_resize = cv2.resize(overlay, (w, int(h * 0.8)))
                    # Overlay the resized image onto the frame
                    frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])

            # Encode the frame as PNG
            ret, buffer = cv2.imencode('.png', frame)
            frame = buffer.tobytes()

            # Yield the frame in a streaming response
            yield (b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)


#second try
from flask import Flask, render_template, request, jsonify, Response
import cv2
import cvzone
import requests
import os
import numpy as np

app = Flask(__name__)

# Global variables to hold the processed PNG path and overlay
glasses_png_path = None
overlay = None

# Function to resize image while maintaining aspect ratio
def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    # Calculate new dimensions maintaining the aspect ratio
    if aspect_ratio > 1:
        # Wider image
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        # Taller image
        new_h = target_height
        new_w = int(target_height * aspect_ratio)

    # Resize image with the calculated dimensions
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a transparent 600x600 image
    final_img = np.zeros((target_height, target_width, 4), dtype=np.uint8)

    # Calculate the padding to center the resized image
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    # Place the resized image on the transparent background
    final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return final_img

# Function to remove the background from an image
def remove_background(image_path, api_key, output_path):
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

        # Resize image while maintaining aspect ratio
        resized_img = resize_with_aspect_ratio(img, 800, 800)

        # Save the resized image
        cv2.imwrite(output_path, resized_img)

        return output_path
    else:
        return None

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global glasses_png_path, overlay
        # Handle image upload and background removal here
        file = request.files['glasses_image']
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['jpg', 'jpeg', 'png']:
            return jsonify({'status': 'error', 'message': 'Please upload a valid image format (jpg, jpeg, png)'})
        
        # Save the uploaded image
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)

        # Call remove.bg API to remove the background
        api_key = 'Q1zzpzfqEq2oxpc7pBkmXJRy'  # Replace with your actual remove.bg API key
        output_path = f'static/output/{os.path.splitext(file.filename)[0]}.png'
        # glasses_png_path = remove_background(file_path, api_key, output_path)

        if glasses_png_path:
            # Load the processed image as overlay
            # overlay = cv2.imread(glasses_png_path, cv2.IMREAD_UNCHANGED)

            return jsonify({'status': 'success', 'image_url': '/' + glasses_png_path})
        else:
            return jsonify({'status': 'error', 'message': 'Background removal failed'})

    return render_template('index2.html')
# glasses_png_path = 'static/output/glass4.png'

# overlay = cv2.imread(glasses_png_path, cv2.IMREAD_UNCHANGED)
def resize_image_for_detection(image, max_width=800):
    h, w = image.shape[:2]
    if w > max_width:
        scale_ratio = max_width / w
        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

@app.route('/webcam_feed')
def webcam_feed():
    def generate_frames():
        global overlay
        cap = cv2.VideoCapture(0)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize image for consistent face detection
            resized_frame = resize_image_for_detection(frame)
            gray_scale = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5)

            if overlay is not None:
                for (x, y, w, h) in faces:
                    # Adjust scaling based on face size
                    glasses_scaling_factor = 1.5 if w < 100 else 1.2  # Dynamically adjust scaling
                    glasses_width = int(w * glasses_scaling_factor)
                    glasses_height = int(glasses_width * 0.6)

                    overlay_resize = cv2.resize(overlay, (glasses_width, glasses_height))

                    y_offset = y + int(h * 0.01)
                    x_offset = x

                    frame = cvzone.overlayPNG(resized_frame, overlay_resize, [x_offset, y_offset])

            ret, buffer = cv2.imencode('.png', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)



# third try with dlib which is good
import cv2
import dlib
import cvzone
import numpy as np
import os
import requests
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# Load dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Path to the .dat file

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

# Function to detect facial landmarks
def get_landmarks(frame):
    # Detect faces using dlib
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

    # For each face, detect the landmarks
    landmarks_list = []
    for face in faces:
        shape = predictor(gray_frame, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        landmarks_list.append(landmarks)
    
    return landmarks_list

# Function to remove the background from an image using remove.bg API
def remove_background(image_path, api_key, output_path):
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
        return None

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global glasses_png_path, overlay
        # Handle image upload and background removal here
        file = request.files['glasses_image']
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['jpg', 'jpeg', 'png']:
            return jsonify({'status': 'error', 'message': 'Please upload a valid image format (jpg, jpeg, png)'})
        
        # Save the uploaded image
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)

        # Call remove.bg API to remove the background
        api_key = 'Q1zzpzfqEq2oxpc7pBkmXJRy'  # Replace with your actual remove.bg API key
        output_path = f'static/output/output.png'
        glasses_png_path = remove_background(file_path, api_key, output_path)

        if glasses_png_path:
            # Load the processed image as overlay
            overlay = cv2.imread(glasses_png_path, cv2.IMREAD_UNCHANGED)

            return jsonify({'status': 'success', 'image_url': '/' + glasses_png_path})
        else:
            return jsonify({'status': 'error', 'message': 'Background removal failed'})

    return render_template('index2.html')


# glasses_png_path = 'static/output/glass.png'

# overlay = cv2.imread(glasses_png_path, cv2.IMREAD_UNCHANGED)
# Route for accessing the webcam and applying the overlay
@app.route('/webcam_feed')
def webcam_feed():
    def generate_frames():
        global overlay
        cap = cv2.VideoCapture(0)  # Open the webcam
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame to a manageable size for face detection
            resized_frame = resize_image_for_detection(frame)
            
            # Get facial landmarks
            landmarks_list = get_landmarks(resized_frame)

            if overlay is not None and landmarks_list:
                for landmarks in landmarks_list:
                    # Check if landmarks for eyes are detected
                    if len(landmarks) >= 68:
                        # Get the coordinates for the eyes (left eye: landmarks 36-41, right eye: landmarks 42-47)
                        left_eye = landmarks[36:42]  # Left eye coordinates
                        right_eye = landmarks[42:48]  # Right eye coordinates

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
                            glasses_width = int(eye_width * 2.6)  # Increase from 2 to 2.8 for larger glasses
                            
                            # Maintain aspect ratio while resizing
                            aspect_ratio = original_glasses_height / original_glasses_width
                            glasses_height = int(glasses_width * aspect_ratio)

                            # Calculate the midpoint between the two eyes to position the glasses
                            eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)

                            # Position the glasses so they are centered on the eye midpoint
                            glasses_x = eye_center[0] - int(glasses_width / 2)
                            glasses_y = eye_center[1] - int(glasses_height / 2)

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

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)

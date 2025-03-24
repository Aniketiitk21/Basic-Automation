import os
import cv2
import uuid
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv8 model
MODEL_PATH = r"C:\Users\mypc\Desktop\Baisc auto\bestmode.pt"
model = YOLO(MODEL_PATH)

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            if file.filename.lower().endswith((".mp4", ".avi", ".mov")):
                # Process Video
                output_video_path = os.path.join(OUTPUT_FOLDER, f"processed_{uuid.uuid4().hex}.mp4")
                process_video(file_path, output_video_path)
                return render_template("video_player.html", video_path=f"/outputs/{os.path.basename(output_video_path)}")

            else:
                # Process Image
                output_image_path = os.path.join(OUTPUT_FOLDER, f"result_{uuid.uuid4().hex}.jpg")
                process_image(file_path, output_image_path)
                return send_file(output_image_path, mimetype="image/jpeg")

    return render_template("index.html")

@app.route("/outputs/<path:filename>")
def serve_output_file(filename):
    """Serves files from the outputs folder."""
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

def process_image(image_path, output_path):
    """Runs YOLO detection on an image and saves the output."""
    image = cv2.imread(image_path)
    results = model(image_path)  # Run inference

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index

            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add label

    cv2.imwrite(output_path, image)

def process_video(video_path, output_path):
    """Processes a video frame by frame with YOLO detection."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)

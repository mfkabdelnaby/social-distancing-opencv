from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename
from src.video_processor import VideoProcessor
import cv2
import numpy as np
import os

app = Flask(__name__)

# The instance of VideoProcessor
video_processor = None

# Define the paths to your YOLO model's configuration and weights files
configPath = "yolo-coco/yolov3.cfg"
weightsPath = "yolo-coco/yolov3.weights"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    global video_processor
    f = request.files["file"]

    # Save the uploaded file to a temporary location
    temp_path = "temp.avi"  # replace with your desired path
    f.save(temp_path)

    # Load the YOLO model
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Initiate the YOLO Model
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Open the temporary video with cv2.VideoCapture
    video = cv2.VideoCapture(temp_path)

    # Initialize the VideoProcessor with the video object
    video_processor = VideoProcessor(net, ln, video)

    return Response(
        process_and_stream(video, temp_path),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def process_and_stream(video, temp_path):
    global video_processor
    while True:
        res, frame = video.read()
        if not res:
            break

        # process the frame
        frame = video_processor.process_frame(frame)

        # convert the image to jpg format
        ret, jpeg = cv2.imencode(".jpg", frame)
        frame = jpeg.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    # Remove the temporary video file after processing is done
    os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

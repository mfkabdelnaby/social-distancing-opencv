# Object Tracking and Social Distancing Detection Using Computer Vision
This project implements object tracking, motion heatmap generation, and social distancing violation detection in video footage. It is primarily built with Python, OpenCV, and YOLO (You Only Look Once) model for object detection.

## Developed Using

- Python 3.10
- OpenCV
- Pandas
- NumPy
- YOLO v3

## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```


## Features
* **Object Tracking:** The project uses the Centroid Tracking algorithm to track objects in video footage. It assigns a unique ID to each object and tracks its movement across frames.

* **Motion Heatmap:** The motion heatmap function highlights areas in the footage where movement is detected, providing a visual representation of motion intensity.

* **Social Distancing Detection:** The project can detect instances where social distancing norms (as per given minimum social distance) are not followed.

* **Bird's Eye View Transformation:** Provides a bird's eye view of the tracked objects for better visualization of their spatial relation.

* **Data Collection:** The system collects data about each tracked object, including entry/exit counts and times, and saves it in CSV files.

* **Performance Metrics:** The system computes and displays the frame processing rate in frames per second (FPS).


## Usage

The main script is `main.py`, which takes in two arguments:
* `input_dir`: The directory containing the input video file(s).
* `output_dir`: The directory where output files will be saved.

You can run the script as follows:
```bash
python main.py <input_dir> <output_dir>
```

## Drawing a Gate Line

This project includes a functionality that allows users to manually draw a line gate. The points from the drawn line are then used in the main function for further calculations.

To use this feature, run the `draw_line.py` script. You will be prompted to manually draw a line gate. After you have drawn the line gate, the points will be saved and used for entries and exits calculations in the project.

```bash
python draw_line.py
```
![ezgif com-video-to-gif](https://github.com/mustafarrag/social-distancing-opencv/assets/39211751/f89a0b29-eaa9-48f8-b2fc-5f4a82cebc91)



## Output

The script generates the following outputs:

1. `out.avi`: Video file with object tracking and social distancing violations highlighted.
2. `out_bird.avi`: Bird's eye view of the tracked objects.
3. `out_heat.avi`: Video with a motion heatmap overlay.
4. `final_heatmap.jpg`: Final heatmap image for the entire video.
5. Dataframes saved as CSV files:
    - `place_df.csv`: Contains information about entries, exits, IDs, and centroids.
    - `user_df.csv`: Contains information about object IDs, centroids, and timestamps.
    - `violations_df.csv`: Contains information about social distancing violations.
  
![ezgif com-video-to-gif (1)](https://github.com/mustafarrag/social-distancing-opencv/assets/39211751/b14bdf1a-6ebd-4392-aabf-f1b09421ab44)


## Web Interface with Flask

This project also includes a simple Flask application that can be used as an interface to interact with the app. The Flask application is designed to receive a video file, process it, and stream the results back to the user.

To run the Flask app, navigate to the project directory and run the following command:

```bash
python app.py
```

This will start the Flask development server. You can access the application by navigating to http://localhost:5000 in your web browser.

**Endpoints**
* **GET /:** This endpoint serves the main page of the application.

* **POST /upload:** This endpoint receives a video file, processes it using the VideoProcessor class, and streams the processed video frames back to the user. The request should include a file in the form data.

The application expects the video to be sent as part of a POST request to the /upload endpoint. The video is then processed frame by frame, and the processed frames are streamed back to the user in a continuous response.


This feature provides an easy-to-use, interactive way for users to leverage the project's capabilities.


## Future Work
This project is in active development and there are several areas targeted for future improvements and new features. Here are a few of them:

* **Improve Line Drawing:** The current project allows for manual line drawing. Future iterations could include an automated system that can detect optimal lines based on certain criteria.
* **Advanced Detection & Tracking:** Upgrades to advanced YOLO versions, such as v5 or v6, are under consideration for better detection. Additionally, the incorporation of the Kalman filter or Hungarian algorithm may enhance trajectory prediction and occlusion handling.

* **Extend Flask Application:** The existing Flask application could be further developed to include features such as heatmaps, bird view outputs, and real-time data visualization.
  
* **Deploy Application:** The application can be potentially deployed on a cloud platform to extend its accessibility to a wider user base.





  







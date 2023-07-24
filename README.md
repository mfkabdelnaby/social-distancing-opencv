# Object Tracking and Social Distancing Detection Using Computer Vision
This project implements object tracking, motion heatmap generation, and social distancing violation detection in video footage. It is primarily built with Python, OpenCV, and YOLO (You Only Look Once) model for object detection.

## Developed Using

- Python 3.10
- OpenCV
- Pandas
- NumPy

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




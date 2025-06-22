# Iris Gaze Tracker

This project is a real-time gaze and head direction tracker that uses a standard webcam. It leverages the MediaPipe Face Landmarker library to identify facial features and employs a machine learning model to estimate the user's gaze point on the screen. The system is designed with a sophisticated calibration process to personalize the model for each user.

![Gaze Tracking Demo](https://youtu.be/O_dZVA1TVNI)

## Key Features

- **Real-Time Face Tracking**: Smoothly tracks the user's face and iris positions using MediaPipe.
- **Background Removal**: Isolates the user's face against a neutral background for reduced distraction and improved focus.
- **Advanced Gaze Estimation**: Predicts the user's on-screen gaze point using a machine learning model.
- **Dynamic Gaze Visualization**: Displays the predicted gaze point as a "cloud" of dots, providing intuitive visual feedback.
- **Adjustable Sensitivity**: Allows real-time control over the gaze tracking responsiveness using the Up and Down arrow keys.
- **Two Calibration Modes**:
    1.  **Click-Based Calibration**: A point-by-point calibration for quick setup.
    2.  **Continuous Calibration**: A more advanced, dynamic calibration where the user follows a moving target, capturing a rich dataset for superior accuracy.

## Technical Evolution & Improvements

The project underwent several key iterations to improve the accuracy and stability of the gaze tracking, moving from a simple proof-of-concept to a robust estimation system.

### 1. Initial Prototype: Simple Iris Centering
The first version of the tracker used a basic approach: calculating the geometric center of the iris landmarks. While functional, this method was highly susceptible to jitter and minor landmark detection inaccuracies.

### 2. Ellipse Fitting for Feature Stability
**Improvement**: Instead of using a simple average for the iris center, we upgraded to fitting an ellipse (`cv2.fitEllipse`) to the 5 iris landmark points.
**Benefit**: This approach provides much richer and more stable features. By extracting the ellipse's center, axis ratio (how circular or flat it is), and orientation angle, the model gets a much better understanding of the iris's shape and tilt, which is a direct proxy for gaze direction. This significantly reduced the jitter caused by minor landmark fluctuations.

### 3. Upgraded Model: `RandomForestRegressor`
**Improvement**: The initial `LinearRegression` model was replaced with a more powerful `RandomForestRegressor`.
**Benefit**: A linear model can only capture straight-line relationships. A Random Forest, being an ensemble of decision trees, can learn the complex, non-linear patterns that truly exist between eye features and screen coordinates. This upgrade allowed the system to make much more accurate predictions based on the high-quality ellipse features.

### 4. Head Pose Normalization (Canonical Transformation)
**Improvement**: This was the most significant architectural upgrade. We leveraged the `facial_transformation_matrixes` from MediaPipe to implement head pose normalization.
**Benefit**: This technique mathematically isolates eye movements from head movements. Before feeding data to the model, we transform the facial landmarks into a "canonical" 3D space, as if the user's head is always facing directly forward. This forces the model to learn the *pure* relationship between the iris's orientation and the gaze point, without the confounding variable of head tilt or camera perspective. The result is a dramatically more robust and accurate system that is resilient to natural head movements.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/iris-gaze-tracker.git
    cd iris-gaze-tracker
    ```

2.  **Create a virtual environment and install dependencies:**
    This project uses `uv` for package management.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt`, you can create one from `pyproject.toml` or manually install `opencv-python`, `mediapipe`, and `scikit-learn`)*

3.  **Download the Face Landmarker Model:**
    Download the `face_landmarker.task` file from [MediaPipe's website](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/index#models) and place it in the root directory of the project.

## How to Run

1.  **Ensure you are in the project directory with the virtual environment activated.**

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **Follow the on-screen calibration instructions.**
    - The application will start in fullscreen mode.
    - A message will prompt you to press **Enter** to begin calibration.
    - Follow the moving percentage number (0% to 100%) with your eyes for about 20 seconds.

4.  **Use the application.**
    - After calibration, the main application will start.
    - The screen will show your face, the fitted ellipses on your irises, and a cloud of dots indicating your predicted gaze point.
    - The following controls are available:
        - **Up Arrow**: Increase the responsiveness (speed) of the gaze cloud.
        - **Down Arrow**: Decrease the responsiveness for smoother movement.
        - **'r'**: Restart the calibration process at any time.
        - **'q'**: Quit the application.


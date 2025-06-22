import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
from types import SimpleNamespace

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Iris landmark indices for MediaPipe Face Landmarker
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Face oval landmark indices
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Face direction reference points
NOSE_TIP = 1
CHIN = 175
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
FOREHEAD = 10

def normalize_landmarks(landmarks, transform_matrix):
    """
    Transforms landmarks to a canonical, front-facing view.
    This removes head rotation and translation from the landmark data.
    """
    # Invert the transformation matrix to go from world space back to the canonical model space
    try:
        inv_transform_matrix = np.linalg.inv(transform_matrix)
    except np.linalg.LinAlgError:
        return None

    normalized_landmarks = []
    for lm in landmarks:
        # Create a 4D homogeneous vector for the landmark
        vec = np.array([lm.x, lm.y, lm.z, 1.0])
        
        # Apply the inverse transformation
        transformed_vec = inv_transform_matrix @ vec
        
        # Normalize by the w-component to get back to 3D coordinates
        if transformed_vec[3] != 0:
            normalized_vec = transformed_vec / transformed_vec[3]
            # Store as a landmark-like object
            normalized_landmarks.append(SimpleNamespace(x=normalized_vec[0], y=normalized_vec[1], z=normalized_vec[2]))
        else:
            # Fallback for safety, though unlikely
            normalized_landmarks.append(SimpleNamespace(x=lm.x, y=lm.y, z=lm.z))

    return normalized_landmarks

def get_iris_features(landmarks, iris_indices, w, h, is_normalized=False, scale_factor=1000):
    """
    Fits an ellipse to the iris landmarks and extracts features.
    Handles both original and normalized landmarks.
    """
    points = []
    for idx in iris_indices:
        if idx < len(landmarks):
            points.append(landmarks[idx])

    if len(points) < 5:
        return None, None

    # For normalized landmarks, we use a constant scaling factor.
    # For original landmarks, we use the frame dimensions.
    if is_normalized:
        pixel_points = np.array([(p.x * scale_factor, p.y * scale_factor) for p in points], dtype=np.int32)
    else:
        pixel_points = np.array([(p.x * w, p.y * h) for p in points], dtype=np.int32)
    
    try:
        # fitEllipse requires at least 5 points
        ellipse = cv2.fitEllipse(pixel_points)
        center, axes, angle = ellipse
        
        # Ensure axes are not zero to avoid division errors
        if axes[0] == 0 or axes[1] == 0:
            return None, None

        # Axis ratio is a key feature for gaze direction
        axis_ratio = min(axes) / max(axes)
        
        return ( (int(center[0]), int(center[1])), axis_ratio, angle), ellipse
    except cv2.error:
        return None, None

def get_face_direction(landmarks, w, h):
    """Calculates face direction (yaw and pitch)."""
    nose_tip = landmarks[NOSE_TIP]
    left_eye = landmarks[LEFT_EYE_CORNER]
    right_eye = landmarks[RIGHT_EYE_CORNER]

    nose_x = int(nose_tip.x * w)
    nose_y = int(nose_tip.y * h)
    left_eye_x = int(left_eye.x * w)
    left_eye_y = int(left_eye.y * h)
    right_eye_x = int(right_eye.x * w)
    right_eye_y = int(right_eye.y * h)

    face_center_x = (left_eye_x + right_eye_x) // 2
    face_center_y = (left_eye_y + right_eye_y) // 2

    horizontal_offset = nose_x - face_center_x
    yaw_direction = "LEFT" if horizontal_offset < -10 else "RIGHT" if horizontal_offset > 10 else "CENTER"

    vertical_offset = nose_y - face_center_y
    pitch_direction = "UP" if vertical_offset < -5 else "DOWN" if vertical_offset > 5 else "LEVEL"
    
    return yaw_direction, pitch_direction

def draw_landmarks(image, landmarks, connections=None, color=(0, 255, 0), thickness=2):
    """Draw landmarks on the image"""
    h, w = image.shape[:2]
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), thickness, color, -1)

def draw_iris(image, landmarks):
    """Draw iris landmarks and the fitted ellipse."""
    h, w = image.shape[:2]
    
    # Draw left iris
    _, left_ellipse = get_iris_features(landmarks, LEFT_IRIS, w, h)
    if left_ellipse:
        cv2.ellipse(image, left_ellipse, (0, 255, 0), 2)
        # Draw the center of the ellipse
        center = (int(left_ellipse[0][0]), int(left_ellipse[0][1]))
        cv2.circle(image, center, 3, (0, 0, 255), -1)

    # Draw right iris
    _, right_ellipse = get_iris_features(landmarks, RIGHT_IRIS, w, h)
    if right_ellipse:
        cv2.ellipse(image, right_ellipse, (0, 255, 0), 2)
        # Draw the center of the ellipse
        center = (int(right_ellipse[0][0]), int(right_ellipse[0][1]))
        cv2.circle(image, center, 3, (0, 0, 255), -1)

def draw_face_direction(image, landmarks):
    """Draw face direction indicators"""
    h, w = image.shape[:2]
    
    # Get key points
    nose_tip = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    left_eye = landmarks[LEFT_EYE_CORNER]
    right_eye = landmarks[RIGHT_EYE_CORNER]
    forehead = landmarks[FOREHEAD]
    
    # Convert to pixel coordinates
    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
    chin_x, chin_y = int(chin.x * w), int(chin.y * h)
    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
    forehead_x, forehead_y = int(forehead.x * w), int(forehead.y * h)
    
    # Draw face center line (nose to chin)
    cv2.line(image, (nose_x, nose_y), (chin_x, chin_y), (0, 255, 255), 2)
    
    # Draw horizontal line (eye to eye)
    cv2.line(image, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y), (0, 255, 255), 2)
    
    # Draw vertical reference line (forehead to chin through nose)
    cv2.line(image, (forehead_x, forehead_y), (chin_x, chin_y), (255, 255, 0), 2)
    
    # Calculate face direction
    yaw_direction, pitch_direction = get_face_direction(landmarks, w, h)

    face_center_x = (left_eye_x + right_eye_x) // 2
    face_center_y = (left_eye_y + right_eye_y) // 2

    # Draw direction arrow from face center
    arrow_length = 50
    horizontal_offset = nose_x - face_center_x
    vertical_offset = nose_y - face_center_y
    arrow_x = face_center_x + int(horizontal_offset * 0.5)
    arrow_y = face_center_y + int(vertical_offset * 0.5)
    cv2.arrowedLine(image, (face_center_x, face_center_y), (arrow_x, arrow_y), (0, 255, 0), 3)
    
    # Display direction text
    cv2.putText(image, f"Yaw: {yaw_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Pitch: {pitch_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw key reference points
    cv2.circle(image, (nose_x, nose_y), 5, (0, 255, 0), -1)  # Nose tip
    cv2.circle(image, (face_center_x, face_center_y), 5, (255, 0, 255), -1)  # Face center

def draw_gaze_cloud(image, center, num_dots=50, spread=35, color=(0, 255, 255), radius=2):
    """Draws a cloud of dots around a center point."""
    if center is None:
        return
    
    points = np.random.normal(loc=center, scale=spread, size=(num_dots, 2)).astype(np.int32)
    
    h, w = image.shape[:2]
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), radius, color, -1)

class GazeEstimator:
    def __init__(self, calibration_data, w, h):
        self.model_x = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_y = RandomForestRegressor(n_estimators=100, random_state=42)
        self.w = w
        self.h = h
        self._train(calibration_data)
    
    def _train(self, calibration_data):
        if not calibration_data or len(calibration_data) < 3:
             print("Not enough calibration data to train gaze model.")
             self.model_x = None
             self.model_y = None
             return
        
        features = []
        labels_x = []
        labels_y = []

        for data in calibration_data:
            # Features are now simpler, as head pose is removed by normalization
            (left_center, left_ratio, left_angle) = data['left_iris_features']
            (right_center, right_ratio, right_angle) = data['right_iris_features']

            features.append([
                left_center[0], left_center[1], left_ratio, left_angle,
                right_center[0], right_center[1], right_ratio, right_angle,
            ])
            labels_x.append(data['target_coords'][0])
            labels_y.append(data['target_coords'][1])
        
        self.model_x.fit(features, labels_x)
        self.model_y.fit(features, labels_y)
        print("Gaze estimation model (RandomForest) trained on NORMALIZED ellipse features.")

    def predict(self, left_iris_features, right_iris_features):
        if not self.model_x or not self.model_y:
            return None

        (left_center, left_ratio, left_angle) = left_iris_features
        (right_center, right_ratio, right_angle) = right_iris_features

        features = np.array([
            left_center[0], left_center[1], left_ratio, left_angle,
            right_center[0], right_center[1], right_ratio, right_angle,
        ]).reshape(1, -1)

        gaze_x = self.model_x.predict(features)[0]
        gaze_y = self.model_y.predict(features)[0]

        return int(gaze_x), int(gaze_y)

class Calibrator:
    def __init__(self, window_name, w, h, num_random_points=10):
        self.window_name = window_name
        self.w = w
        self.h = h
        margin = 100
        self.points = [
            (w // 2, h // 2),
            (margin, margin),
            (w - margin, margin),
            (w - margin, h - margin),
            (margin, h - margin)
        ]
        self.point_names = ["CENTER", "TOP LEFT", "TOP RIGHT", "BOTTOM RIGHT", "BOTTOM LEFT"]

        # Add random calibration points for more robust training
        for i in range(num_random_points):
            rand_x = np.random.randint(margin, w - margin)
            rand_y = np.random.randint(margin, h - margin)
            self.points.append((rand_x, rand_y))
            self.point_names.append(f"RANDOM {i + 1}")

        self.current_point_index = 0
        self.click_received = False
        self.calibration_data = []

    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            target_pos = self.points[self.current_point_index]
            # Check if click is within the target's radius
            if np.linalg.norm(np.array([x, y]) - np.array(target_pos)) < 25:
                self.click_received = True

    def draw_target(self, image, center):
        radius = 20
        cv2.circle(image, center, radius, (0, 0, 255), 2)
        cv2.line(image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 0, 255), 2)
        cv2.line(image, (center[0], center[1] - radius), (center[0], center[1] + radius), (0, 0, 255), 2)

    def run(self, cap, detector):
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        print("Starting calibration process. Press 'q' to quit or 'r' to restart.")

        for i, point in enumerate(self.points):
            self.current_point_index = i
            self.click_received = False
            print(f"Please look at the {self.point_names[i]} target and click on it.")
            
            while not self.click_received:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame. Exiting calibration.")
                    return None

                # Display target on a clean background
                background_color = (30, 30, 30)
                output_frame = np.full((self.h, self.w, 3), background_color, np.uint8)
                self.draw_target(output_frame, point)

                # Display calibration progress
                progress_display_text = f"Point {i + 1} of {len(self.points)}"
                cv2.putText(output_frame, progress_display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow(self.window_name, output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Calibration aborted by user.")
                    return None
                elif key == ord('r'): # Allow restart during calibration
                    return "restart"

            # --- Data collection with averaging ---
            print("Click registered. Hold still, collecting data...")
            n_samples = 30
            collected_data = []

            for _ in range(n_samples):
                ret, frame = cap.read()
                if not ret: continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)
                
                if detection_result.face_landmarks and detection_result.facial_transformation_matrixes:
                    # Normalize landmarks to remove head pose
                    transform_matrix = detection_result.facial_transformation_matrixes[0]
                    landmarks = detection_result.face_landmarks[0]
                    normalized_landmarks = normalize_landmarks(landmarks, transform_matrix)
                    if not normalized_landmarks: continue

                    # Get features from the *normalized* landmarks
                    left_iris_features, _ = get_iris_features(normalized_landmarks, LEFT_IRIS, self.w, self.h, is_normalized=True)
                    right_iris_features, _ = get_iris_features(normalized_landmarks, RIGHT_IRIS, self.w, self.h, is_normalized=True)
                    
                    if left_iris_features and right_iris_features:
                        collected_data.append({
                            'left_iris_features': left_iris_features,
                            'right_iris_features': right_iris_features,
                        })

                # Display feedback to user
                feedback_frame = np.full((self.h, self.w, 3), (30, 30, 30), np.uint8)
                self.draw_target(feedback_frame, point)
                progress_text = f"Collecting data: {len(collected_data)}/{n_samples}"
                
                # Calculate text size to center it on the target
                (text_width, text_height), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = point[0] - text_width // 2
                text_y = point[1] + text_height // 2
                
                cv2.putText(feedback_frame, progress_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(self.window_name, feedback_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     return None

            if not collected_data:
                print(f"Could not detect face/iris for {self.point_names[i]}. Skipping point.")
                continue

            # --- Average the collected features for a more stable reading ---
            def average_features(feature_list, index):
                 return sum(f[index] for f in feature_list) / len(feature_list)
            
            avg_left_center_x = average_features([d['left_iris_features'][0] for d in collected_data], 0)
            avg_left_center_y = average_features([d['left_iris_features'][0] for d in collected_data], 1)
            avg_left_ratio = average_features([d['left_iris_features'] for d in collected_data], 1)
            avg_left_angle = average_features([d['left_iris_features'] for d in collected_data], 2)

            avg_right_center_x = average_features([d['right_iris_features'][0] for d in collected_data], 0)
            avg_right_center_y = average_features([d['right_iris_features'][0] for d in collected_data], 1)
            avg_right_ratio = average_features([d['right_iris_features'] for d in collected_data], 1)
            avg_right_angle = average_features([d['right_iris_features'] for d in collected_data], 2)

            data = {
                'target': self.point_names[i],
                'target_coords': point,
                'left_iris_features': ((avg_left_center_x, avg_left_center_y), avg_left_ratio, avg_left_angle),
                'right_iris_features': ((avg_right_center_x, avg_right_center_y), avg_right_ratio, avg_right_angle),
            }
            self.calibration_data.append(data)
            print(f"CALIBRATION CAPTURED for {self.point_names[i]}:")
            print(f"  - Avg Left Iris Features: Ratio={avg_left_ratio:.2f}, Angle={avg_left_angle:.2f}")
            print(f"  - Avg Right Iris Features: Ratio={avg_right_ratio:.2f}, Angle={avg_right_angle:.2f}")

        cv2.setMouseCallback(self.window_name, lambda *args: None)
        print("\nCalibration complete.")
        return self.calibration_data

class ContinuousCalibrator:
    def __init__(self, window_name, w, h, duration_seconds=20):
        self.window_name = window_name
        self.w = w
        self.h = h
        self.duration_seconds = duration_seconds
        self.fps = 30  # Assumed frame rate for path generation
        self.calibration_data = []

    def _create_path_generator(self):
        """Creates a generator that yields (x, y, percentage) for the target path."""
        total_frames = self.duration_seconds * self.fps
        margin = 100
        
        # Define path segments
        path_points = [
            (margin, margin),
            (self.w - margin, margin),
            (self.w - margin, self.h - margin),
            (margin, self.h - margin),
            (margin, margin),  # Back to start
            (self.w - margin, self.h - margin), # Diagonal
            (self.w // 2, self.h // 2),
            (margin, self.h - margin), # Other diagonal
            (self.w - margin, margin),
        ]

        # Calculate frames per segment
        distances = [np.linalg.norm(np.array(path_points[i+1]) - np.array(path_points[i])) for i in range(len(path_points)-1)]
        total_distance = sum(distances)
        frames_per_segment = [(d / total_distance) * total_frames for d in distances]

        frame_count = 0
        for i in range(len(path_points) - 1):
            start_point = np.array(path_points[i])
            end_point = np.array(path_points[i+1])
            num_frames_for_segment = int(frames_per_segment[i])
            
            for j in range(num_frames_for_segment):
                if frame_count >= total_frames: break
                
                # Linear interpolation between points
                alpha = j / num_frames_for_segment
                current_pos = start_point + alpha * (end_point - start_point)
                percentage = (frame_count / total_frames) * 100
                
                yield int(current_pos[0]), int(current_pos[1]), percentage
                frame_count += 1
        
        # Ensure it runs for the full duration
        while frame_count < total_frames:
            percentage = (frame_count / total_frames) * 100
            yield self.w // 2, self.h // 2, percentage
            frame_count += 1


    def run(self, cap, detector):
        # --- Add instruction screen ---
        while True:
            background_color = (30, 30, 30)
            output_frame = np.full((self.h, self.w, 3), background_color, np.uint8)
            
            line1 = "Keep your eyes following the moving number."
            line2 = "Press Enter to start calibration."
            line3 = "Press 'q' to quit."

            # Calculate text sizes to center them
            (l1_w, l1_h), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            (l2_w, l2_h), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            (l3_w, l3_h), _ = cv2.getTextSize(line3, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            cv2.putText(output_frame, line1, (self.w // 2 - l1_w // 2, self.h // 2 - l1_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(output_frame, line2, (self.w // 2 - l2_w // 2, self.h // 2 + l2_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(output_frame, line3, (self.w // 2 - l3_w // 2, self.h // 2 + l2_h + l3_h * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(self.window_name, output_frame)
            
            # Wait indefinitely for a key press
            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # Enter key
                break
            elif key == ord('q'):
                print("Calibration aborted by user.")
                return None
        
        print("Starting continuous calibration. Follow the moving number with your eyes.")
        print("Press 'q' to quit.")

        path_generator = self._create_path_generator()
        
        for x, y, percentage in path_generator:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting calibration.")
                return None
            
            # --- Perform detection and collect data ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks and detection_result.facial_transformation_matrixes:
                transform_matrix = detection_result.facial_transformation_matrixes[0]
                landmarks = detection_result.face_landmarks[0]
                normalized_landmarks = normalize_landmarks(landmarks, transform_matrix)

                if normalized_landmarks:
                    left_features, _ = get_iris_features(normalized_landmarks, LEFT_IRIS, self.w, self.h, is_normalized=True)
                    right_features, _ = get_iris_features(normalized_landmarks, RIGHT_IRIS, self.w, self.h, is_normalized=True)
                    
                    if left_features and right_features:
                        data = {
                            'target_coords': (x, y),
                            'left_iris_features': left_features,
                            'right_iris_features': right_features,
                        }
                        self.calibration_data.append(data)

            # --- Display the target ---
            background_color = (30, 30, 30)
            output_frame = np.full((self.h, self.w, 3), background_color, np.uint8)
            text = f"{int(percentage)}%"
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            text_x = x - text_width // 2
            text_y = y + text_height // 2
            
            cv2.putText(output_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            cv2.imshow(self.window_name, output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Calibration aborted by user.")
                return None
        
        if not self.calibration_data:
            print("Calibration failed: No data was collected.")
            return None

        print(f"\nContinuous calibration complete. Collected {len(self.calibration_data)} data points.")
        return self.calibration_data

def show_webcam_feed():
    """
    Captures video from the default webcam and displays it with face landmarks,
    iris detection, and face direction indicators.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam feed started. Press 'q' to quit.")

    window_name = 'Face Landmarker - Iris & Direction'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get frame dimensions for calibrator
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame for calibration.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    h, w, _ = frame.shape
    
    calibration_data = None
    while calibration_data is None:
        # To use the old click-based calibrator, swap the next two lines
        # calibrator = Calibrator(window_name, w, h)
        calibrator = ContinuousCalibrator(window_name, w, h, duration_seconds=20)
        calibration_data = calibrator.run(cap, detector)
        if calibration_data == "restart":
            print("Restarting calibration...")
            calibration_data = None
            # To use the old click-based calibrator, swap the next two lines
            # calibrator = Calibrator(window_name, w, h)
            calibrator = ContinuousCalibrator(window_name, w, h, duration_seconds=20)
            calibration_data = calibrator.run(cap, detector)
            if calibration_data == "restart":
                print("Restarting calibration...")
                calibration_data = None
        elif calibration_data is None:
            print("Calibration was not completed. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return

    gaze_estimator = GazeEstimator(calibration_data, w, h)
    print("\nStarting main application...")

    gaze_speed = 0.03 # Smoothing factor: higher is faster, more responsive
    smoothed_gaze_point = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Create a solid color background
        h, w, _ = frame.shape
        background_color = (30, 30, 30)  # BGR for #1e1e1e
        output_frame = np.full((h, w, 3), background_color, np.uint8)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face landmarks
        detection_result = detector.detect(mp_image)
        
        # Process detection results
        if detection_result.face_landmarks and detection_result.facial_transformation_matrixes:
            transform_matrix = detection_result.facial_transformation_matrixes[0]
            landmarks = detection_result.face_landmarks[0]
            normalized_landmarks = normalize_landmarks(landmarks, transform_matrix)

            if normalized_landmarks:
                left_iris_features, _ = get_iris_features(normalized_landmarks, LEFT_IRIS, w, h, is_normalized=True)
                right_iris_features, _ = get_iris_features(normalized_landmarks, RIGHT_IRIS, w, h, is_normalized=True)

                if left_iris_features and right_iris_features:
                    gaze_point = gaze_estimator.predict(left_iris_features, right_iris_features)
                    
                    if gaze_point:
                        if smoothed_gaze_point is None:
                            smoothed_gaze_point = gaze_point
                        else:
                            # Apply exponential moving average for smoothing
                            sx = gaze_speed * gaze_point[0] + (1 - gaze_speed) * smoothed_gaze_point[0]
                            sy = gaze_speed * gaze_point[1] + (1 - gaze_speed) * smoothed_gaze_point[1]
                            smoothed_gaze_point = (int(sx), int(sy))

                        draw_gaze_cloud(output_frame, smoothed_gaze_point)
                        
                        if smoothed_gaze_point:
                            text = f"Gaze: ({smoothed_gaze_point[0]}, {smoothed_gaze_point[1]})"
                            cv2.putText(output_frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw annotations on the final frame
                draw_iris(output_frame, landmarks)
                draw_face_direction(output_frame, landmarks)
        
        # Display speed control info
        speed_text = f"Speed: {gaze_speed:.2f} (Up/Down arrow to change)"
        cv2.putText(output_frame, speed_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, output_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('u'):
            gaze_speed = min(1.0, gaze_speed + 0.01)
        elif key == ord('d'):
            gaze_speed = max(0.01, gaze_speed - 0.01)
        elif key == ord('r'):
            print("Restarting calibration...")
            calibration_data = None
            while calibration_data is None:
                 calibrator = ContinuousCalibrator(window_name, w, h, duration_seconds=20)
                 calibration_data = calibrator.run(cap, detector)
                 if calibration_data == "restart":
                     print("Restarting calibration...")
                     calibration_data = None
            if calibration_data:
                gaze_estimator = GazeEstimator(calibration_data, w, h)
                print("Calibration completed successfully!")
            else:
                print("Calibration failed or was cancelled.")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped and resources released.")

if __name__ == '__main__':
    show_webcam_feed()
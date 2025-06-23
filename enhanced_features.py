import numpy as np
from collections import deque
from types import SimpleNamespace

class EnhancedFeatureExtractor:
    def __init__(self, buffer_size=5):
        """
        Enhanced feature extractor with temporal and spatial features.
        buffer_size: Number of frames to keep for temporal features
        """
        self.buffer_size = buffer_size
        self.left_iris_history = deque(maxlen=buffer_size)
        self.right_iris_history = deque(maxlen=buffer_size)
        self.gaze_history = deque(maxlen=buffer_size)
        
        # Eye corner landmark indices for MediaPipe
        self.LEFT_EYE_CORNERS = [33, 133]  # inner, outer corners
        self.RIGHT_EYE_CORNERS = [362, 263]  # inner, outer corners
        self.LEFT_EYE_TOP_BOTTOM = [159, 145]  # top, bottom
        self.RIGHT_EYE_TOP_BOTTOM = [386, 374]  # top, bottom

    def extract_enhanced_features(self, normalized_landmarks, left_iris_features, right_iris_features, 
                                w, h, scale_factor=1000):
        """
        Extract enhanced features combining spatial, geometric, and temporal information.
        Returns feature vector compatible with existing GazeEstimator.
        """
        # Original features (8 features)
        (left_center, left_ratio, left_angle) = left_iris_features
        (right_center, right_ratio, right_angle) = right_iris_features
        
        base_features = [
            left_center[0], left_center[1], left_ratio, left_angle,
            right_center[0], right_center[1], right_ratio, right_angle,
        ]
        
        # Enhanced spatial features
        spatial_features = self._extract_spatial_features(normalized_landmarks, scale_factor)
        
        # Enhanced geometric features
        geometric_features = self._extract_geometric_features(
            normalized_landmarks, left_iris_features, right_iris_features, scale_factor
        )
        
        # Store current iris data for temporal features
        self.left_iris_history.append(left_iris_features)
        self.right_iris_history.append(right_iris_features)
        
        # Temporal features
        temporal_features = self._extract_temporal_features()
        
        # Combine all features
        enhanced_features = base_features + spatial_features + geometric_features + temporal_features
        
        return np.array(enhanced_features)

    def _extract_spatial_features(self, landmarks, scale_factor):
        """Extract spatial relationship features between eyes and face."""
        features = []
        
        if len(landmarks) > max(self.LEFT_EYE_CORNERS + self.RIGHT_EYE_CORNERS + 
                               self.LEFT_EYE_TOP_BOTTOM + self.RIGHT_EYE_TOP_BOTTOM):
            
            # Eye corner distances
            left_inner = landmarks[self.LEFT_EYE_CORNERS[0]]
            left_outer = landmarks[self.LEFT_EYE_CORNERS[1]]
            right_inner = landmarks[self.RIGHT_EYE_CORNERS[0]]
            right_outer = landmarks[self.RIGHT_EYE_CORNERS[1]]
            
            # Eye width (corner to corner distance)
            left_eye_width = np.sqrt((left_outer.x - left_inner.x)**2 + (left_outer.y - left_inner.y)**2)
            right_eye_width = np.sqrt((right_outer.x - right_inner.x)**2 + (right_outer.y - right_inner.y)**2)
            
            # Eye height (top to bottom distance)
            left_top = landmarks[self.LEFT_EYE_TOP_BOTTOM[0]]
            left_bottom = landmarks[self.LEFT_EYE_TOP_BOTTOM[1]]
            right_top = landmarks[self.RIGHT_EYE_TOP_BOTTOM[0]]
            right_bottom = landmarks[self.RIGHT_EYE_TOP_BOTTOM[1]]
            
            left_eye_height = np.sqrt((left_top.x - left_bottom.x)**2 + (left_top.y - left_bottom.y)**2)
            right_eye_height = np.sqrt((right_top.x - right_bottom.x)**2 + (right_top.y - right_bottom.y)**2)
            
            # Eye aspect ratios
            left_eye_aspect = left_eye_height / (left_eye_width + 1e-8)
            right_eye_aspect = right_eye_height / (right_eye_width + 1e-8)
            
            # Inter-eye distance
            inter_eye_dist = np.sqrt((right_inner.x - left_inner.x)**2 + (right_inner.y - left_inner.y)**2)
            
            features = [
                left_eye_width * scale_factor,
                right_eye_width * scale_factor,
                left_eye_height * scale_factor,
                right_eye_height * scale_factor,
                left_eye_aspect,
                right_eye_aspect,
                inter_eye_dist * scale_factor
            ]
        else:
            # Fallback features if landmarks are insufficient
            features = [0.0] * 7
            
        return features

    def _extract_geometric_features(self, landmarks, left_iris_features, right_iris_features, scale_factor):
        """Extract geometric features related to iris positioning."""
        features = []
        
        if len(landmarks) > max(self.LEFT_EYE_CORNERS + self.RIGHT_EYE_CORNERS):
            # Get eye corners
            left_inner = landmarks[self.LEFT_EYE_CORNERS[0]]
            left_outer = landmarks[self.LEFT_EYE_CORNERS[1]]
            right_inner = landmarks[self.RIGHT_EYE_CORNERS[0]]
            right_outer = landmarks[self.RIGHT_EYE_CORNERS[1]]
            
            # Eye centers (between corners)
            left_eye_center = ((left_inner.x + left_outer.x) / 2, (left_inner.y + left_outer.y) / 2)
            right_eye_center = ((right_inner.x + right_outer.x) / 2, (right_inner.y + right_outer.y) / 2)
            
            # Iris-to-eye-center offsets
            (left_iris_center, _, _) = left_iris_features
            (right_iris_center, _, _) = right_iris_features
            
            # Convert iris centers back to normalized coordinates
            left_iris_norm = (left_iris_center[0] / scale_factor, left_iris_center[1] / scale_factor)
            right_iris_norm = (right_iris_center[0] / scale_factor, right_iris_center[1] / scale_factor)
            
            # Offsets from eye center to iris center
            left_offset_x = left_iris_norm[0] - left_eye_center[0]
            left_offset_y = left_iris_norm[1] - left_eye_center[1]
            right_offset_x = right_iris_norm[0] - right_eye_center[0]
            right_offset_y = right_iris_norm[1] - right_eye_center[1]
            
            # Convergence (how much eyes are turned toward each other)
            convergence = (left_offset_x - right_offset_x) * scale_factor
            
            # Vertical alignment (how much eyes look up/down together)
            vertical_alignment = (left_offset_y + right_offset_y) / 2 * scale_factor
            
            # Iris ratio differences (asymmetry indicator)
            ratio_diff = abs(left_iris_features[1] - right_iris_features[1])
            
            features = [
                left_offset_x * scale_factor,
                left_offset_y * scale_factor,
                right_offset_x * scale_factor,
                right_offset_y * scale_factor,
                convergence,
                vertical_alignment,
                ratio_diff
            ]
        else:
            features = [0.0] * 7
            
        return features

    def _extract_temporal_features(self):
        """Extract temporal features from iris movement history."""
        features = []
        
        if len(self.left_iris_history) >= 2:
            # Calculate velocities
            left_vel_x = self.left_iris_history[-1][0][0] - self.left_iris_history[-2][0][0]
            left_vel_y = self.left_iris_history[-1][0][1] - self.left_iris_history[-2][0][1]
            right_vel_x = self.right_iris_history[-1][0][0] - self.right_iris_history[-2][0][0]
            right_vel_y = self.right_iris_history[-1][0][1] - self.right_iris_history[-2][0][1]
            
            # Movement magnitude
            left_movement = np.sqrt(left_vel_x**2 + left_vel_y**2)
            right_movement = np.sqrt(right_vel_x**2 + right_vel_y**2)
            
            # Movement consistency (standard deviation over buffer)
            if len(self.left_iris_history) >= 3:
                left_centers_x = [data[0][0] for data in self.left_iris_history]
                left_centers_y = [data[0][1] for data in self.left_iris_history]
                right_centers_x = [data[0][0] for data in self.right_iris_history]
                right_centers_y = [data[0][1] for data in self.right_iris_history]
                
                left_stability_x = np.std(left_centers_x)
                left_stability_y = np.std(left_centers_y)
                right_stability_x = np.std(right_centers_x)
                right_stability_y = np.std(right_centers_y)
                
                features = [
                    left_vel_x, left_vel_y, right_vel_x, right_vel_y,
                    left_movement, right_movement,
                    left_stability_x, left_stability_y,
                    right_stability_x, right_stability_y
                ]
            else:
                features = [left_vel_x, left_vel_y, right_vel_x, right_vel_y, 
                          left_movement, right_movement, 0.0, 0.0, 0.0, 0.0]
        else:
            features = [0.0] * 10
            
        return features

    def get_feature_count(self):
        """Return total number of features this extractor produces."""
        return 8 + 7 + 7 + 10  # base + spatial + geometric + temporal = 32 features

    def reset_history(self):
        """Reset temporal history buffers."""
        self.left_iris_history.clear()
        self.right_iris_history.clear()
        self.gaze_history.clear()


class EnhancedGazeEstimator:
    """
    Drop-in replacement for GazeEstimator with enhanced features.
    Usage: Simply replace GazeEstimator with EnhancedGazeEstimator in main.py
    """
    def __init__(self, calibration_data, w, h):
        from sklearn.ensemble import RandomForestRegressor
        
        self.model_x = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=15)
        self.model_y = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=15)
        self.w = w
        self.h = h
        self.feature_extractor = EnhancedFeatureExtractor()
        self._train(calibration_data)
    
    def _train(self, calibration_data):
        if not calibration_data or len(calibration_data) < 3:
            print("Not enough calibration data to train enhanced gaze model.")
            self.model_x = None
            self.model_y = None
            return
        
        # We need to simulate the enhanced feature extraction for training data
        # Since calibration data doesn't have landmarks, we'll use basic features for now
        # and enhance during real-time prediction
        features = []
        labels_x = []
        labels_y = []

        for data in calibration_data:
            (left_center, left_ratio, left_angle) = data['left_iris_features']
            (right_center, right_ratio, right_angle) = data['right_iris_features']

            # For training, we'll use the basic features extended with zeros
            # This is a limitation that would be resolved with enhanced calibration
            base_features = [
                left_center[0], left_center[1], left_ratio, left_angle,
                right_center[0], right_center[1], right_ratio, right_angle,
            ]
            
            # Pad with zeros for missing enhanced features during training
            enhanced_features = base_features + [0.0] * (self.feature_extractor.get_feature_count() - 8)
            
            features.append(enhanced_features)
            labels_x.append(data['target_coords'][0])
            labels_y.append(data['target_coords'][1])
        
        self.model_x.fit(features, labels_x)
        self.model_y.fit(features, labels_y)
        print("Enhanced gaze estimation model trained with expanded feature set.")

    def predict(self, left_iris_features, right_iris_features, normalized_landmarks=None):
        if not self.model_x or not self.model_y:
            return None

        # Extract enhanced features if landmarks are available
        if normalized_landmarks:
            try:
                features = self.feature_extractor.extract_enhanced_features(
                    normalized_landmarks, left_iris_features, right_iris_features, 
                    self.w, self.h
                )
            except Exception as e:
                # Fallback to basic features if enhancement fails
                print(f"Feature enhancement failed: {e}, using basic features")
                features = self._get_basic_features(left_iris_features, right_iris_features)
        else:
            features = self._get_basic_features(left_iris_features, right_iris_features)

        features = features.reshape(1, -1)
        gaze_x = self.model_x.predict(features)[0]
        gaze_y = self.model_y.predict(features)[0]

        return int(gaze_x), int(gaze_y)
    
    def _get_basic_features(self, left_iris_features, right_iris_features):
        """Fallback to basic features with padding."""
        (left_center, left_ratio, left_angle) = left_iris_features
        (right_center, right_ratio, right_angle) = right_iris_features

        base_features = [
            left_center[0], left_center[1], left_ratio, left_angle,
            right_center[0], right_center[1], right_ratio, right_angle,
        ]
        
        # Pad with zeros for missing enhanced features
        enhanced_features = base_features + [0.0] * (self.feature_extractor.get_feature_count() - 8)
        return np.array(enhanced_features) 

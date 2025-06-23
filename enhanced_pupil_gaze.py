from threshold_pupil_detection import detect_pupil_threshold, extract_eye_region

# Add these eye region landmark indices to your main.py
LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def get_pupil_features(landmarks, eye_indices, image, is_left=True):
    """
    Replace get_iris_features with pupil detection
    """
    # Extract eye region
    eye_region_data = extract_eye_region(landmarks, eye_indices, image)
    if not eye_region_data:
        return None
    
    eye_region, offset = eye_region_data
    
    # Detect pupil
    pupil_center, pupil_size = detect_pupil_threshold(eye_region)
    
    if pupil_center:
        # Convert back to full image coordinates
        global_x = pupil_center[0] + offset[0]
        global_y = pupil_center[1] + offset[1]
        
        # Get eye corner positions for normalization
        if is_left:
            inner_corner = landmarks[33]  # Left eye inner corner
            outer_corner = landmarks[133]  # Left eye outer corner
        else:
            inner_corner = landmarks[362]  # Right eye inner corner
            outer_corner = landmarks[263]  # Right eye outer corner
        
        h, w = image.shape[:2]
        inner_x, inner_y = int(inner_corner.x * w), int(inner_corner.y * h)
        outer_x, outer_y = int(outer_corner.x * w), int(outer_corner.y * h)
        
        # Calculate relative position (0-1 scale)
        eye_width = abs(outer_x - inner_x)
        eye_height = abs(outer_y - inner_y)
        
        if eye_width > 0:
            relative_x = (global_x - min(inner_x, outer_x)) / eye_width
            relative_y = (global_y - min(inner_y, outer_y)) / eye_height if eye_height > 0 else 0.5
        else:
            relative_x = 0.5
            relative_y = 0.5
        
        # Return features: (center_x, center_y), relative_position, pupil_size
        return ((global_x, global_y), (relative_x, relative_y), pupil_size)
    
    return None 

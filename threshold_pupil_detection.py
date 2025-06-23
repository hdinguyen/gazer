import cv2

def detect_pupil_threshold(eye_region):
    """
    Simple threshold-based pupil detection
    Most effective for your use case
    """
    # Convert to grayscale
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to isolate dark pupil
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (likely pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get center and area
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(largest_contour)
            return (cx, cy), area
    
    return None, None

def extract_eye_region(landmarks, eye_indices, image, padding=20):
    """
    Extract eye region from face landmarks
    """
    h, w = image.shape[:2]
    
    # Get eye landmark coordinates
    eye_points = []
    for idx in eye_indices:
        if idx < len(landmarks):
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            eye_points.append((x, y))
    
    if not eye_points:
        return None
    
    # Get bounding box
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]
    
    x_min = max(0, min(xs) - padding)
    x_max = min(w, max(xs) + padding)
    y_min = max(0, min(ys) - padding)
    y_max = min(h, max(ys) + padding)
    
    # Extract region
    eye_region = image[y_min:y_max, x_min:x_max]
    return eye_region, (x_min, y_min)  # region and offset 

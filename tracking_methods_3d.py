import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

def coarse_find(frame, min_size=(150, 150)):
    """Detect eye region using Haar cascade"""
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=min_size)
    return eyes

def remove_bright_spots(image, threshold=200, replace=0):
    """Replace bright pixels (reflections) with darker value"""
    mask = image < threshold
    image[~mask] = replace
    return image, mask

def find_dark_area(image):
    """Find darkest region in image using grid search"""
    num_grids = 9
    h, w = image.shape[:2]
    grid_h = h // num_grids
    grid_w = w // num_grids 
    darkest_val = 255
    darkest_square = None
    for i in range(num_grids):
        for j in range(num_grids):
            grid = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            mean_val = np.mean(grid)
            if mean_val < darkest_val:
                darkest_val = mean_val
                darkest_square = (i*grid_h, j*grid_w, grid_h, grid_w)
    return darkest_square, darkest_val
    
def threshold_images(image, dark_point, thresholds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50]):
    """Create multiple thresholded images with morphological operations"""
    images = []
    h, w = image.shape
    denoised = cv2.GaussianBlur(image, (5, 5), 0)   
    kernel = np.ones((3, 3), np.uint8)

    for t in thresholds:
        _, binary = cv2.threshold(denoised, dark_point + t, 255, cv2.THRESH_BINARY_INV)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill holes using flood fill
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flood = opened.copy()
        cv2.floodFill(flood, mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(opened, flood_inv)
        
        # Smooth edges
        filled = cv2.GaussianBlur(filled, (5, 5), 2)
        images.append(filled)

    return images

def get_contours(images, min_area=1500, margin=3):
    """Extract largest contour from each thresholded image"""
    filtered_contours = []
    contour_images = []

    for img in images:
        h, w = img.shape[:2]
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and edge proximity
        kept = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            pts = cnt.reshape(-1, 2)
            if (pts[:,0] < margin).any() or (pts[:,0] > w - margin).any() \
            or (pts[:,1] < margin).any() or (pts[:,1] > h - margin).any():
                continue
            kept.append(cnt)
        
        # Keep only largest contour
        kept = sorted(kept, key=cv2.contourArea, reverse=True)[:1]
        if len(kept) > 0:
            all_pts = np.vstack([c.reshape(-1,2) for c in kept])
            all_pts = all_pts.reshape(-1,1,2).astype(np.int32)
        else:
            all_pts = np.array([], dtype=np.int32).reshape(-1,1,2)

        # Compute convex hull
        hull = cv2.convexHull(all_pts)

        ci = np.zeros_like(img)
        cv2.drawContours(ci, hull, -1, 255, 2)
        filtered_contours.append(kept)
        contour_images.append(ci)

    return filtered_contours, contour_images

def fit_ellipse(contour, bias_factor=3):
    """Fit ellipse with bottom bias and vertical reflection"""
    pts = np.vstack([c.reshape(-1,2) for c in contour])
    
    # Add bottom bias (weight lower points more heavily)
    mean_y = np.mean(pts[:,1])
    bottom_pts = pts[pts[:,1] > mean_y]
    
    if bottom_pts.size and bias_factor > 0:
        weighted_pts = np.concatenate([pts] + [bottom_pts]*bias_factor, axis=0)
    else:
        weighted_pts = pts
    
    weighted_pts = weighted_pts.reshape(-1,1,2).astype(np.int32)
    if len(weighted_pts) < 5:
        return None
    
    temp = cv2.fitEllipse(weighted_pts)
    if temp is None:
        return None
    
    # Add vertical reflection if ellipse is taller than wide
    (cx, cy), (w, h), ang = temp
    if w < h:
        reflected_pts = weighted_pts.copy()
        reflected_pts[:, 0, 1] = 2 * cy - reflected_pts[:, 0, 1]
        weighted_pts = np.concatenate([weighted_pts, reflected_pts], axis=0)

    return cv2.fitEllipse(weighted_pts)

def check_flip(ellipse):
    """Normalize ellipse angle to [-90, 90) range"""
    (cx, cy), (w, h), ang = ellipse

    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180

    if ang == 90:
        ang = 0
        w, h = h, w
    return (cx, cy), (w, h), ang

def process_eye_crop(frame, eyes):
    """Extract and preprocess eye region"""
    x, y, w, h = eyes[0]
    size = max(w, h)
    eye_crop = frame[y:y+size, x:x+size].copy()
    eye_crop, _ = remove_bright_spots(eye_crop, threshold=220, replace=100)
    eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    return eye_gray, x, y, size

def generate_ellipse_candidates(eye_gray, dark_val, thresholds):
    """Generate ellipse candidates using custom and OpenCV fitting"""
    thresholded_images = threshold_images(eye_gray, dark_val, thresholds=thresholds)
    contours, contour_images = get_contours(thresholded_images)
    ellipse_images = []
    ellipses_custom = []
    ellipses_opencv = []
    
    for cnt_list in contours:
        temp_img = eye_gray.copy()

        if len(cnt_list) == 0:
            ellipses_custom.append(None)
            ellipses_opencv.append(None)
            ellipse_images.append(temp_img)
            continue
        
        # Fit ellipses using both methods
        box_custom = fit_ellipse(cnt_list)
        box_opencv = cv2.fitEllipse(cnt_list[0]) if len(cnt_list[0]) >= 5 else None
        
        # Draw ellipses for visualization
        if box_custom is not None:
            cv2.ellipse(temp_img, box_custom, 0, 2)
        if box_opencv is not None:
            cv2.ellipse(temp_img, box_opencv, 255, 1)
        
        ellipses_custom.append(box_custom)
        ellipses_opencv.append(box_opencv)
        ellipse_images.append(temp_img)
    
    return thresholded_images, contour_images, ellipse_images, ellipses_custom, ellipses_opencv

def calculate_ellipse_scores(thresholded_images, ellipses):
    """Score ellipses based on shape and pixel distribution"""
    N = len(thresholded_images) 
    percents = []
    
    for i in range(N):
        eye_thresh = thresholded_images[i]
        ellipse = ellipses[i]
        
        if ellipse is None:
            percents.append(0)
            continue
        
        # Filter by aspect ratio (must be roughly circular)
        ellipse_ratio = ellipse[1][1] / ellipse[1][0]
        if ellipse_ratio > 2 or ellipse_ratio < 0.8:
            percents.append(0)
            continue

        # Create ellipse mask
        mask = np.zeros_like(eye_thresh)
        (cx, cy), (w, h), ang = ellipse
        cv2.ellipse(mask, (int(cx), int(cy)), (int(w/2), int(h/2)), ang, 0, 360, 255, -1)

        # Calculate inside ratio (white pixels inside ellipse)
        inside_total = cv2.countNonZero(mask)
        inside_white = cv2.countNonZero(cv2.bitwise_and(eye_thresh, mask))
        inside_ratio = inside_white / inside_total if inside_total > 0 else 0

        # Calculate outside ratio (black pixels outside ellipse)
        outside_mask = cv2.bitwise_not(mask)
        outside_total = cv2.countNonZero(outside_mask)
        outside_black = cv2.countNonZero(cv2.bitwise_and(cv2.bitwise_not(eye_thresh), outside_mask))
        outside_ratio = outside_black / outside_total if outside_total > 0 else 0

        # Combine ratios into final score
        percent = ((inside_ratio + outside_ratio * 0.25) / 1.5)
        percents.append(percent)
    
    return percents

def select_best_ellipse(ellipses, percents, prev_ellipse, x, y, frame_idx):
    """Select best ellipse with fallback to previous frame if needed"""
    best_idx = int(np.argmax(percents))
    best_ellipse = ellipses[best_idx]

    # Fallback to previous ellipse if current is None
    if best_ellipse is None:
        if prev_ellipse is not None:
            best_ellipse, prev_x, prev_y = prev_ellipse
            x, y = prev_x, prev_y
        else:
            return None, x, y

    # Validate against previous ellipse
    if prev_ellipse is not None:
        (pcx, pcy), (pw, ph), pang = prev_ellipse[0]
        (cx, cy), (w, h), ang = best_ellipse
        
        # Check for teleporting (sudden large movement)
        if (w*h) < (pw*ph) and (abs(cy - pcy) > 100 or abs(cx - pcx) > 100):
            best_ellipse = prev_ellipse[0]
            x, y = prev_ellipse[1], prev_ellipse[2]
        elif (w*h) < 0.3 * (pw*ph):
            best_ellipse = prev_ellipse[0]

    return best_ellipse, x, y

def apply_smoothing(best_ellipse, x, y, ema, x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha):
    """Apply exponential moving average smoothing to ellipse parameters"""
    best_ellipse = check_flip(best_ellipse)
    (cx, cy), (w, h), ang = best_ellipse

    # Convert to full frame coordinates
    current = np.array([cx + x, cy + y, w, h, ang], dtype=np.float32)
    
    if ema is None:
        return best_ellipse, current.copy()
    
    # Apply EMA with different alphas for each parameter
    alphas = np.array([x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha], dtype=np.float32)
    ema = alphas * current + (1.0 - alphas) * ema

    sm_cx, sm_cy, sm_w, sm_h, sm_ang = ema
    full_ellipse = ((float(sm_cx), float(sm_cy)), (float(sm_w), float(sm_h)), float(sm_ang))

    return full_ellipse, ema

def draw_orthogonal_ray(image, ellipse, length=100, color=(0, 255, 0), thickness=1):
    """Draw ray perpendicular to ellipse surface"""
    (cx, cy), (major_axis, minor_axis), angle = ellipse
    
    angle_rad = np.deg2rad(angle)
    normal_dx = (minor_axis / 2) * np.cos(angle_rad)
    normal_dy = (minor_axis / 2) * np.sin(angle_rad)

    pt1 = (int(cx - length * normal_dx / (minor_axis / 2)), int(cy - length * normal_dy / (minor_axis / 2)))
    pt2 = (int(cx + length * normal_dx / (minor_axis / 2)), int(cy + length * normal_dy / (minor_axis / 2)))

    cv2.line(image, pt1, pt2, color, thickness)
    return image

def display_results(frame, thresholded_images, contour_images, ellipse_images, 
                    full_ellipse_custom, ellipse_opencv, cx, cy, x, y, frame_idx, thresholds, best_idx):
    """Display processing steps and results"""
    N = len(thresholded_images)
    H, W = thresholded_images[0].shape
    grid = np.zeros((3 * H, N * W), dtype=np.uint8)
    
    # Create grid: threshold, contour, ellipse rows
    for i in range(N):
        grid[0:H, i*W:(i+1)*W] = thresholded_images[i]
        grid[H:2*H, i*W:(i+1)*W] = contour_images[i]
        grid[2*H:3*H, i*W:(i+1)*W] = ellipse_images[i]
    
    grid_disp = cv2.resize(grid, (1400, 600))
    
    # Label thresholds and highlight selected
    col_width = 1400 // N
    for i in range(N):
        label = f"+{thresholds[i]}"
        color = 255 if i == best_idx else 128
        cv2.putText(grid_disp, label, (i * col_width + 5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2 if i == best_idx else 1)
        
        if i == best_idx:
            x_start = i * col_width
            x_end = (i + 1) * col_width
            cv2.rectangle(grid_disp, (x_start, 0), (x_end - 1, 600), 200, 2)
    
    cv2.imshow("Threshold | Contour | Ellipse", grid_disp)
    
    # Draw custom ellipse (green)
    cv2.ellipse(frame, full_ellipse_custom, (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    
    # Display custom ellipse parameters
    (ccx, ccy), (cw, ch), cang = full_ellipse_custom
    cv2.putText(frame, f"Ang:{cang:.1f}", (int(ccx) + 10, int(ccy) - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, f"Min:{min(cw, ch):.1f}", (int(ccx) + 10, int(ccy) + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw OpenCV ellipse (cyan)
    if ellipse_opencv is not None:
        (ocx, ocy), (ow, oh), oang = ellipse_opencv
        opencv_full = ((ocx + x, ocy + y), (ow, oh), oang)
        cv2.ellipse(frame, opencv_full, (255, 255, 0), 1)
        cv2.circle(frame, (int(ocx + x), int(ocy + y)), 2, (255, 0, 255), -1)
        draw_orthogonal_ray(frame, opencv_full, length=150, color=(255, 255, 0), thickness=1)
        
        cv2.putText(frame, f"Ang:{oang:.1f}", (int(ocx + x) - 60, int(ocy + y) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"Min:{min(ow, oh):.1f}", (int(ocx + x) - 60, int(ocy + y) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    cv2.putText(frame, f"Frame: {frame_idx} | Green=Custom | Cyan=OpenCV", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Eye Tracking", frame)

def select_video_file():
    """Open file dialog to select video"""
    root = tk.Tk()
    root.withdraw()
    
    filetypes = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
        ("All files", "*.*")
    ]
    
    filename = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
    root.destroy()
    return filename

def main():
    video_path = select_video_file()
    
    # Smoothing parameters
    x_alpha = 0.75
    y_alpha = 0.75
    width_alpha = 0.5
    height_alpha = 0.1
    rotation_alpha = 1.0
    
    # Threshold values to try
    thresholds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    
    prev_ellipse = None
    ema = None
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    prev_eyes = None
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use top half of frame
        frame = frame[:frame.shape[0] // 2, :]

        # Detect eye region every 5 frames
        if frame_idx % 5 == 0:
            eyes = coarse_find(frame)
            if len(eyes) > 0:
                prev_eyes = eyes.copy()
            elif prev_eyes is not None:
                eyes = prev_eyes
            else:
                frame_idx += 1
                continue

        # Process eye region
        eye_gray, x, y, size = process_eye_crop(frame, eyes)
        dark_square, dark_val = find_dark_area(eye_gray)
        
        # Generate ellipse candidates
        thresholded_images, contour_images, ellipse_images, ellipses_custom, ellipses_opencv = \
            generate_ellipse_candidates(eye_gray, dark_val, thresholds)
        
        # Score ellipses based on OpenCV fits
        percents = calculate_ellipse_scores(thresholded_images, ellipses_opencv)
        
        # Select best ellipse
        current_x, current_y = x, y
        best_ellipse_custom, ellipse_x, ellipse_y = select_best_ellipse(
            ellipses_custom, percents, prev_ellipse, x, y, frame_idx)
        
        if best_ellipse_custom is None:
            frame_idx += 1
            continue
        
        # Get corresponding OpenCV ellipse
        best_idx = int(np.argmax(percents))
        best_ellipse_opencv = ellipses_opencv[best_idx]
        
        prev_ellipse = (best_ellipse_custom, ellipse_x, ellipse_y)

        # Apply smoothing
        full_ellipse_custom, ema = apply_smoothing(
            best_ellipse_custom, ellipse_x, ellipse_y, ema,
            x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha)
        
        (cx, cy), (w, h), ang = full_ellipse_custom

        # Display results
        display_results(frame, thresholded_images, contour_images, ellipse_images, 
                       full_ellipse_custom, best_ellipse_opencv, cx, cy, 
                       current_x, current_y, frame_idx, thresholds, best_idx)
        frame_idx += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    
    end_time = time.time()
    print(f"\nProcessed {frame_idx} frames in {end_time - start_time:.2f} seconds.")
    print(f"Average FPS: {frame_idx / (end_time - start_time):.2f}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

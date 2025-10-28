import cv2
import numpy as np
import time
import random
import tkinter as tk
from tkinter import filedialog

try:
    from EyeTracker import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")


class PupilDetector:
    def __init__(self, window_name=""):
        # Window identifier for display
        self.window_name = window_name
        
        # Sphere center tracking state
        self.ray_lines = []
        self.model_centers = []
        self.max_rays = 100
        self.prev_model_center_avg = (320, 240)
        self.max_observed_distance = 100  # Initial estimate, will be updated from 3D triangulation
        
        # Smoothing state
        self.prev_ellipse_custom = None
        self.prev_ellipse_opencv = None
        self.ema = None
        
        # Smoothing parameters
        self.x_alpha = 0.75
        self.y_alpha = 0.75
        self.width_alpha = 0.5
        self.height_alpha = 0.1
        self.rotation_alpha = 1.0
        
        # Threshold values
        self.thresholds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
        
        # Eye detection state
        self.prev_eyes = None
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def coarse_find(self, frame, min_size=(150, 150)):
        """Detect eye region using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=min_size)
        return eyes
    
    def remove_bright_spots(self, image, threshold=200, replace=0):
        """Replace bright pixels (reflections) with darker value"""
        mask = image < threshold
        image[~mask] = replace
        return image, mask
    
    def find_dark_area(self, image):
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
    
    def threshold_images(self, image, dark_point, thresholds):
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
    
    def get_contours(self, images, min_area=1500, margin=3):
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
    
    def fit_ellipse(self, contour, bias_factor=3):
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
    
    def check_flip(self, ellipse):
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
    
    def process_eye_crop(self, frame, eyes):
        """Extract and preprocess eye region"""
        x, y, w, h = eyes[0]
        size = max(w, h)
        eye_crop = frame[y:y+size, x:x+size].copy()
        eye_crop, _ = self.remove_bright_spots(eye_crop, threshold=220, replace=100)
        eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        return eye_gray, x, y, size
    
    def generate_ellipse_candidates(self, eye_gray, dark_val):
        """Generate ellipse candidates using custom and OpenCV fitting"""
        thresholded_images = self.threshold_images(eye_gray, dark_val, thresholds=self.thresholds)
        contours, contour_images = self.get_contours(thresholded_images)
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
            box_custom = self.fit_ellipse(cnt_list)
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
    
    def calculate_ellipse_scores(self, thresholded_images, ellipses):
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
    
    def select_best_ellipse(self, ellipses_custom, ellipses_opencv, percents, x, y, frame_idx):
        """Select best ellipse with fallback to previous frame if needed"""
        best_idx = int(np.argmax(percents))
        # print(f"Best index: {best_idx} - Percent: {percents[best_idx]}")
        best_ellipse_custom = ellipses_custom[best_idx]
        best_ellipse_opencv = ellipses_opencv[best_idx]

        # Fallback to previous ellipse if current is None
        if best_ellipse_custom is None:
            if self.prev_ellipse_custom is not None:
                return self.prev_ellipse_custom, self.prev_ellipse_opencv

            else:
                return None

        # Validate against previous ellipse
        if self.prev_ellipse_custom is not None:
            prev_ellipse_data, prev_x, prev_y = self.prev_ellipse_custom
            (pcx, pcy), (pw, ph), pang = prev_ellipse_data
            (cx, cy), (w, h), ang = best_ellipse_custom
            
            # Check for teleporting (sudden large movement)
            if (w*h) < (0.8*pw*ph) and (abs(cy - pcy) > 100 or abs(cx - pcx) > 100):
                return self.prev_ellipse_custom, self.prev_ellipse_opencv

        return [best_ellipse_custom, x, y], [best_ellipse_opencv, x, y]
    
    def apply_smoothing(self, ellipse_data):
        """Apply exponential moving average smoothing to ellipse parameters"""
        best_ellipse, x, y = ellipse_data
        best_ellipse = self.check_flip(best_ellipse)
        (cx, cy), (w, h), ang = best_ellipse

        # Convert to full frame coordinates
        current = np.array([cx + x, cy + y, w, h, ang], dtype=np.float32)
        
        if self.ema is None:
            self.ema = current.copy()
            return best_ellipse
        
        # Apply EMA with different alphas for each parameter
        alphas = np.array([self.x_alpha, self.y_alpha, self.width_alpha, 
                          self.height_alpha, self.rotation_alpha], dtype=np.float32)
        self.ema = alphas * current + (1.0 - alphas) * self.ema

        sm_cx, sm_cy, sm_w, sm_h, sm_ang = self.ema
        full_ellipse = ((float(sm_cx), float(sm_cy)), (float(sm_w), float(sm_h)), float(sm_ang))

        return full_ellipse
    
    def find_line_intersection(self, ellipse1, ellipse2):
        """Compute intersection of two lines orthogonal to ellipse surfaces"""
        (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
        (cx2, cy2), (_, minor_axis2), angle2 = ellipse2

        angle1_rad = np.deg2rad(angle1)
        angle2_rad = np.deg2rad(angle2)

        # Direction vectors for the two lines
        dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
        dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)

        # Line equations: (cx1, cy1) + t1 * (dx1, dy1) = (cx2, cy2) + t2 * (dx2, dy2)
        A = np.array([[dx1, -dx2], [dy1, -dy2]])
        B = np.array([cx2 - cx1, cy2 - cy1])

        if np.linalg.det(A) == 0:
            return None  # Lines are parallel

        t1, t2 = np.linalg.solve(A, B)

        intersection_x = cx1 + t1 * dx1
        intersection_y = cy1 + t1 * dy1

        return (int(intersection_x), int(intersection_y))
    
    def compute_average_intersection(self, frame, N):
        """Select N random rays, compute their intersections, and return average"""
        if len(self.ray_lines) < 2 or N < 2:
            return None

        height, width = frame.shape[:2]
        selected_lines = random.sample(self.ray_lines, min(N, len(self.ray_lines)))
        intersections = []

        # Compute intersections for consecutive pairs
        for i in range(len(selected_lines) - 1):
            line1 = selected_lines[i]
            line2 = selected_lines[i + 1]

            angle1 = line1[2]
            angle2 = line2[2]

            if abs(angle1 - angle2) >= 2:  # Ensure at least 2 degree difference
                intersection = self.find_line_intersection(line1, line2)
                
                if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                    intersections.append(intersection)

        if not intersections:
            return None

        avg_x = np.mean([pt[0] for pt in intersections])
        avg_y = np.mean([pt[1] for pt in intersections])

        return (int(avg_x), int(avg_y))
    
    def update_and_average_point(self, new_point, N):
        """Maintain a list of N most recent points and return their average"""
        self.model_centers.append(new_point)

        if len(self.model_centers) > N:
            self.model_centers.pop(0)

        if not self.model_centers:
            return None

        avg_x = int(np.mean([p[0] for p in self.model_centers]))
        avg_y = int(np.mean([p[1] for p in self.model_centers]))

        return (avg_x, avg_y)
    
    def display_threshold_grid(self, thresholded_images, contour_images, ellipse_images, best_idx, frame_idx, scores):
        """Display threshold processing grid"""
        N = len(thresholded_images)
        H, W = thresholded_images[0].shape
        grid = np.zeros((3 * H, N * W), dtype=np.uint8)
        
        # Create grid: threshold, contour, ellipse rows
        for i in range(N):
            grid[0:H, i*W:(i+1)*W] = thresholded_images[i]
            grid[H:2*H, i*W:(i+1)*W] = contour_images[i]
            # Convert ellipse images to grayscale if needed
            if len(ellipse_images[i].shape) == 3:
                grid[2*H:3*H, i*W:(i+1)*W] = cv2.cvtColor(ellipse_images[i], cv2.COLOR_BGR2GRAY)
            else:
                grid[2*H:3*H, i*W:(i+1)*W] = ellipse_images[i]
        
        grid_disp = cv2.resize(grid, (1400, 400))
        
        # Label thresholds, scores, and highlight selected
        col_width = 1400 // N
        for i in range(N):
            threshold_label = f"+{self.thresholds[i]}"
            score_label = f"{scores[i]:.2f}"
            color = 255 if i == best_idx else 128
            
            # Draw threshold value at top
            cv2.putText(grid_disp, threshold_label, (i * col_width + 5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2 if i == best_idx else 1)
            
            # Draw score at bottom
            cv2.putText(grid_disp, score_label, (i * col_width + 5, 395), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if i == best_idx:
                x_start = i * col_width
                x_end = (i + 1) * col_width
                cv2.rectangle(grid_disp, (x_start, 0), (x_end - 1, 400), 200, 2)
        
        cv2.imshow(f"Threshold | Contour | Ellipse {self.window_name}", grid_disp)
    
    def track_frame(self, frame, frame_idx):
        """Process a single frame and return tracking results"""
        # Detect eye region every 5 frames
        if frame_idx % 5 == 0:
            eyes = self.coarse_find(frame)
            if len(eyes) > 0:
                self.prev_eyes = eyes.copy()
            elif self.prev_eyes is not None:
                eyes = self.prev_eyes
            else:
                return None
        else:
            eyes = self.prev_eyes
            if eyes is None:
                return None

        # Process eye region
        eye_gray, x, y, size = self.process_eye_crop(frame, eyes)
        dark_square, dark_val = self.find_dark_area(eye_gray)
        
        # Generate ellipse candidates
        thresholded_images, contour_images, ellipse_images, ellipses_custom, ellipses_opencv = \
            self.generate_ellipse_candidates(eye_gray, dark_val)
        
        # Score ellipses based on OpenCV fits
        percents = self.calculate_ellipse_scores(thresholded_images, ellipses_opencv)
        
        # Select best ellipse (returns [ellipse, x, y])
        result = self.select_best_ellipse(ellipses_custom, ellipses_opencv, percents, x, y, frame_idx)
        
        if result is None:
            print("No ellipse found")
            return None
        
        best_ellipse_custom, best_ellipse_opencv = result
        
        # Get corresponding OpenCV ellipse and bundle with crop coordinates
        best_idx = int(np.argmax(percents))
        raw_ellipse = ellipses_opencv[best_idx]
        raw_ellipse = [raw_ellipse, x, y] if raw_ellipse is not None else None
        
        self.prev_ellipse_custom = best_ellipse_custom
        self.prev_ellipse_opencv = best_ellipse_opencv

        # Apply smoothing (pass bundled data)
        full_ellipse_custom = self.apply_smoothing(best_ellipse_custom)
        
        (cx, cy), (w, h), ang = full_ellipse_custom

        # Compute sphere center from ray intersections
        if best_ellipse_opencv is not None:
            ellipse_opencv_data, ex, ey = best_ellipse_opencv
            # Convert to full frame coordinates for ray storage
            (ocx, ocy), (ow, oh), oang = ellipse_opencv_data
            full_ellipse_opencv = ((ocx + ex, ocy + ey), (ow, oh), oang)
            
            # Store ray for sphere center calculation
            self.ray_lines.append(full_ellipse_opencv)
            
            # Prune rays if exceeding max
            if len(self.ray_lines) > self.max_rays:
                self.ray_lines = self.ray_lines[-self.max_rays:]
        
        # Compute sphere center
        model_center_average = (320, 240)
        model_center = self.compute_average_intersection(frame, 5)
        
        if model_center is not None:
            model_center_average = self.update_and_average_point(model_center, 500)
        
        # Fallback to previous center if still at default
        if model_center_average[0] == 320:
            model_center_average = self.prev_model_center_avg
        if model_center_average[0] != 0:
            self.prev_model_center_avg = model_center_average
        
        # Display threshold grid inside the class
        self.display_threshold_grid(thresholded_images, contour_images, ellipse_images, best_idx, frame_idx, percents)
        
        return {
            'full_ellipse_custom': full_ellipse_custom,
            'best_ellipse_opencv': best_ellipse_opencv,
            'raw_ellipse': raw_ellipse,
            'cx': cx,
            'cy': cy,
            'model_center_average': model_center_average,
            'best_idx': best_idx,
            'x': x,
            'y': y
        }


def draw_orthogonal_ray(image, ellipse, length=100, color=(0, 255, 0), thickness=1):
    """Draw ray perpendicular to ellipse surface"""
    (cx, cy), (major_axis, minor_axis), angle = ellipse
    
    angle_rad = np.deg2rad(angle)
    normal_dx = (minor_axis / 2) * np.cos(angle_rad)
    normal_dy = (minor_axis / 2) * np.sin(angle_rad)

    pt1 = (int(cx - length * normal_dx / (minor_axis / 2)), int(cy - length * normal_dy / (minor_axis / 2)))
    pt2 = (int(cx + length * normal_dx / (minor_axis / 2)), int(cy + length * normal_dy / (minor_axis / 2)))

    cv2.line(image, pt1, pt2, color, thickness)


def visualize_circle_2d(circle_3d, baseline_mm=36.442):
    """Visualize 3D circle in X-Z plane (top-down view)"""
    if circle_3d is None:
        return None
    
    center = circle_3d['center']
    left_normals = circle_3d.get('left_normals', (None, None))
    right_normals = circle_3d.get('right_normals', (None, None))
    
    # Find which pair of vectors are closest (true gaze direction)
    left_n1, left_n2 = left_normals
    right_n1, right_n2 = right_normals
    
    # Calculate dot products (closer to 1 = more similar)
    if left_n1 is not None and right_n1 is not None:
        dot_11 = np.dot(left_n1, right_n1)
        dot_12 = np.dot(left_n1, right_n2) if right_n2 is not None else -1
        dot_21 = np.dot(left_n2, right_n1) if left_n2 is not None else -1
        dot_22 = np.dot(left_n2, right_n2) if left_n2 is not None and right_n2 is not None else -1
        
        # Find maximum dot product (closest pair)
        max_dot = max(dot_11, dot_12, dot_21, dot_22)
        
        # Determine which are the true vectors
        left_1_is_true = (max_dot == dot_11 or max_dot == dot_12)
        left_2_is_true = (max_dot == dot_21 or max_dot == dot_22)
        right_1_is_true = (max_dot == dot_11 or max_dot == dot_21)
        right_2_is_true = (max_dot == dot_12 or max_dot == dot_22)
    else:
        # Default: assume all are non-true
        left_1_is_true = False
        left_2_is_true = False
        right_1_is_true = False
        right_2_is_true = False
    
    # Create a blank canvas (500x500 pixels)
    canvas_size = 500
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    # Scale: 1mm = 5 pixels, centered at (250, 450)
    scale = 5.0
    origin_x = canvas_size // 2
    origin_z = canvas_size - 50
    
    def world_to_canvas(x, z):
        """Convert world coordinates (mm) to canvas pixels"""
        canvas_x = int(origin_x + x * scale)
        canvas_z = int(origin_z - z * scale)
        return (canvas_x, canvas_z)
    
    # Draw coordinate axes
    cv2.line(canvas, world_to_canvas(-50, 0), world_to_canvas(50, 0), (100, 100, 100), 1)  # X axis
    cv2.line(canvas, world_to_canvas(0, 0), world_to_canvas(0, 80), (100, 100, 100), 1)    # Z axis
    cv2.putText(canvas, "X", world_to_canvas(48, -3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(canvas, "Z", world_to_canvas(3, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Draw cameras
    cam_left_pos = world_to_canvas(-baseline_mm / 2, 0)
    cam_right_pos = world_to_canvas(baseline_mm / 2, 0)
    cv2.circle(canvas, cam_left_pos, 5, (255, 255, 255), -1)
    cv2.circle(canvas, cam_right_pos, 5, (255, 255, 255), -1)
    cv2.putText(canvas, "L", (cam_left_pos[0] - 15, cam_left_pos[1] + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, "R", (cam_right_pos[0] + 10, cam_right_pos[1] + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw pupil center
    center_pos = world_to_canvas(center[0], center[2])
    cv2.circle(canvas, center_pos, 4, (0, 0, 255), -1)
    
    # Draw viewing rays from cameras to pupil
    cv2.line(canvas, cam_left_pos, center_pos, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(canvas, cam_right_pos, center_pos, (150, 150, 150), 1, cv2.LINE_AA)
    
    # Draw normal vector candidates as lines through center
    vector_length = 15  # mm
    
    # Left camera normals (2 candidates)
    if left_n1 is not None:
        start_pos = world_to_canvas(center[0] - left_n1[0] * vector_length, 
                                    center[2] - left_n1[2] * vector_length)
        end_pos = world_to_canvas(center[0] + left_n1[0] * vector_length, 
                                 center[2] + left_n1[2] * vector_length)
        thickness = 2 if left_1_is_true else 1
        cv2.line(canvas, start_pos, end_pos, (0, 0, 255), thickness)  # BLUE
    
    if left_n2 is not None:
        start_pos = world_to_canvas(center[0] - left_n2[0] * vector_length, 
                                    center[2] - left_n2[2] * vector_length)
        end_pos = world_to_canvas(center[0] + left_n2[0] * vector_length, 
                                 center[2] + left_n2[2] * vector_length)
        thickness = 2 if left_2_is_true else 1
        cv2.line(canvas, start_pos, end_pos, (0, 0, 200), thickness)  # LIGHT BLUE
    
    # Right camera normals (2 candidates)
    if right_n1 is not None:
        start_pos = world_to_canvas(center[0] - right_n1[0] * vector_length, 
                                    center[2] - right_n1[2] * vector_length)
        end_pos = world_to_canvas(center[0] + right_n1[0] * vector_length, 
                                 center[2] + right_n1[2] * vector_length)
        thickness = 3 if right_1_is_true else 1
        cv2.line(canvas, start_pos, end_pos, (0, 255, 0), thickness)  # GREEN
    
    if right_n2 is not None:
        start_pos = world_to_canvas(center[0] - right_n2[0] * vector_length, 
                                    center[2] - right_n2[2] * vector_length)
        end_pos = world_to_canvas(center[0] + right_n2[0] * vector_length, 
                                 center[2] + right_n2[2] * vector_length)
        thickness = 3 if right_2_is_true else 1
        cv2.line(canvas, start_pos, end_pos, (0, 200, 0), thickness)  # DARK GREEN
    
    # Add info text
    info_text = [
        f"Position: ({center[0]:.1f}, {center[2]:.1f}) mm",
        "Left normals: YELLOW/ORANGE",
        "Right normals: GREEN (bright/dark)"
    ]
    for i, text in enumerate(info_text):
        cv2.putText(canvas, text, (10, 20 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return canvas


def display_results(frame, result_top, result_bottom, frame_idx, sphere_radius_top=150, sphere_radius_bottom=150, pupil_3d=None, circle_3d=None):
    """Display eye tracking results for both stereo views on the full frame"""
    height = frame.shape[0]
    half_height = height // 2
    
    # Draw top result
    if result_top is not None:
        full_ellipse_custom = result_top['full_ellipse_custom']
        best_ellipse_opencv = result_top['best_ellipse_opencv']
        raw_ellipse = result_top['raw_ellipse']
        cx = result_top['cx']
        cy = result_top['cy']
        model_center_average = result_top['model_center_average']
        
        # Draw custom ellipse (green)
        cv2.ellipse(frame, full_ellipse_custom, (0, 255, 0), 1)
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        
        # Display custom ellipse parameters
        (ccx, ccy), (cw, ch), cang = full_ellipse_custom
        cv2.putText(frame, f"Ang:{cang:.1f}", (int(ccx) + 10, int(ccy) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"Min:{min(cw, ch):.1f}", (int(ccx) + 10, int(ccy) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if raw_ellipse is not None:
            raw_ellipse_data, rx, ry = raw_ellipse
            (rcx, rcy), (rw, rh), rang = raw_ellipse_data
            raw_full = ((rcx + rx, rcy + ry), (rw, rh), rang)
            cv2.ellipse(frame, raw_full, (255, 255, 0), 1)
        
        # Draw OpenCV ellipse (magenta)
        if best_ellipse_opencv is not None:
            ellipse_opencv_data, ex, ey = best_ellipse_opencv
            (ocx, ocy), (ow, oh), oang = ellipse_opencv_data
            opencv_full = ((ocx + ex, ocy + ey), (ow, oh), oang)
            cv2.ellipse(frame, opencv_full, (255, 0, 255), 1)
            
            # Draw orthogonal ray
            draw_orthogonal_ray(frame, opencv_full, length=50, color=(255, 0, 255), thickness=1)
            
            cv2.putText(frame, f"Ang:{oang:.1f}", (int(ocx + ex) - 60, int(ocy + ey) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(frame, f"Min:{min(ow, oh):.1f}", (int(ocx + ex) - 60, int(ocy + ey) + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw sphere center and radius
        if model_center_average is not None:
            cv2.circle(frame, model_center_average, sphere_radius_top, (255, 50, 50), 2)
            cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)
            
            if cx is not None and cy is not None:
                cv2.line(frame, model_center_average, (int(cx), int(cy)), (255, 150, 50), 2)
    
    # Draw bottom result (offset by half_height)
    if result_bottom is not None:
        full_ellipse_custom = result_bottom['full_ellipse_custom']
        best_ellipse_opencv = result_bottom['best_ellipse_opencv']
        raw_ellipse = result_bottom['raw_ellipse']
        cx = result_bottom['cx']
        cy = result_bottom['cy']
        model_center_average = result_bottom['model_center_average']
        
        # Offset all coordinates by half_height
        (ccx, ccy), (cw, ch), cang = full_ellipse_custom
        full_ellipse_custom_offset = ((ccx, ccy + half_height), (cw, ch), cang)
        
        # Draw custom ellipse (green)
        cv2.ellipse(frame, full_ellipse_custom_offset, (0, 255, 0), 1)
        cv2.circle(frame, (int(cx), int(cy + half_height)), 3, (0, 0, 255), -1)
        
        # Display custom ellipse parameters
        cv2.putText(frame, f"Ang:{cang:.1f}", (int(ccx) + 10, int(ccy + half_height) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"Min:{min(cw, ch):.1f}", (int(ccx) + 10, int(ccy + half_height) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if raw_ellipse is not None:
            raw_ellipse_data, rx, ry = raw_ellipse
            (rcx, rcy), (rw, rh), rang = raw_ellipse_data
            raw_full = ((rcx + rx, rcy + ry + half_height), (rw, rh), rang)
            cv2.ellipse(frame, raw_full, (255, 255, 0), 1)
        
        # Draw OpenCV ellipse (magenta)
        if best_ellipse_opencv is not None:
            ellipse_opencv_data, ex, ey = best_ellipse_opencv
            (ocx, ocy), (ow, oh), oang = ellipse_opencv_data
            opencv_full = ((ocx + ex, ocy + ey + half_height), (ow, oh), oang)
            cv2.ellipse(frame, opencv_full, (255, 0, 255), 1)
            
            # Draw orthogonal ray
            draw_orthogonal_ray(frame, opencv_full, length=50, color=(255, 0, 255), thickness=1)
            
            cv2.putText(frame, f"Ang:{oang:.1f}", (int(ocx + ex) - 60, int(ocy + ey + half_height) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(frame, f"Min:{min(ow, oh):.1f}", (int(ocx + ex) - 60, int(ocy + ey + half_height) + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw sphere center and radius (offset)
        if model_center_average is not None:
            model_center_offset = (model_center_average[0], model_center_average[1] + half_height)
            cv2.circle(frame, model_center_offset, sphere_radius_bottom, (255, 50, 50), 2)
            cv2.circle(frame, model_center_offset, 8, (255, 255, 0), -1)
            
            if cx is not None and cy is not None:
                cv2.line(frame, model_center_offset, (int(cx), int(cy + half_height)), (255, 150, 50), 2)
    
    # Display 3D position if available
    if pupil_3d is not None:
        x_mm, y_mm, z_mm = pupil_3d
        cv2.putText(frame, f"Frame: {frame_idx} | 3D: ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Eye Tracking", frame)
    
    # Display 3D circle visualization
    if circle_3d is not None:
        viz = visualize_circle_2d(circle_3d)
        if viz is not None:
            cv2.imshow("3D Pupil Circle (X-Z view)", viz)


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


def ellipse_to_orthonormal_vectors(ellipse):
    """
    Generate two 3D normal vector candidates from an ellipse
    
    A circle viewed at an angle appears as an ellipse.
    The normal to the circle plane can be calculated from the ellipse parameters.
    
    Parameters:
    - ellipse: ((cx, cy), (w, h), angle) - OpenCV ellipse format
    
    Returns:
    - normal_1, normal_2: Two candidate 3D unit normal vectors to the circle plane
                          (ambiguity in which way the circle is tilted)
    """
    if ellipse is None:
        return None, None
    
    (cx, cy), (w, h), angle = ellipse
    
    # Get semi-axes
    semi_major = max(w, h) / 2
    semi_minor = min(w, h) / 2
    
    # Calculate tilt angle from axis ratio
    # For a circle viewed at angle theta: semi_minor = r * cos(theta)
    if semi_major > 0:
        cos_tilt = semi_minor / semi_major
        cos_tilt = np.clip(cos_tilt, 0, 1)
        tilt_angle = np.arccos(cos_tilt)
    else:
        return None, None
    
    # Convert ellipse angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Determine major axis direction in image plane
    if w >= h:
        # w is the major axis
        major_dir_2d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    else:
        # h is the major axis (90 degrees from angle)
        major_dir_2d = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    
    # The view direction is straight into the image (0, 0, 1) in camera coordinates
    # Viewing direction: +Z into the image
    view_direction = np.array([0, 0, 1])
    
    # Tilt direction in 3D (perpendicular to view, in direction of major axis)
    # Map 2D image direction to 3D: X stays X, Y stays Y, Z=0
    tilt_direction_3d = np.array([major_dir_2d[0], major_dir_2d[1], 0])
    tilt_direction_3d = tilt_direction_3d / np.linalg.norm(tilt_direction_3d)
    
    # Rotate view_direction around tilt_direction by +/- tilt_angle
    # Using Rodrigues' rotation formula
    cos_t = np.cos(tilt_angle)
    sin_t = np.sin(tilt_angle)
    
    cross_prod = np.cross(tilt_direction_3d, view_direction)
    dot_prod = np.dot(tilt_direction_3d, view_direction)
    
    # Two candidates
    normal_1 = view_direction * cos_t + cross_prod * sin_t + tilt_direction_3d * dot_prod * (1 - cos_t)
    normal_1 = normal_1 / np.linalg.norm(normal_1)
    
    normal_2 = view_direction * cos_t - cross_prod * sin_t + tilt_direction_3d * dot_prod * (1 - cos_t)
    normal_2 = normal_2 / np.linalg.norm(normal_2)
    
    return normal_1, normal_2


def camera_normal_to_world(normal_camera, pupil_3d, is_left_camera, baseline_mm=36.442):
    """
    Transform normal vector from camera coordinates to world coordinates
    
    Parameters:
    - normal_camera: (x, y, z) normal in camera frame where Z points into image
    - pupil_3d: (x, y, z) 3D position of pupil in world coordinates
    - is_left_camera: True if left camera, False if right
    - baseline_mm: distance between cameras
    
    Returns:
    - normal_world: (x, y, z) normal in world coordinates
    """
    x_3d, y_3d, z_3d = pupil_3d
    
    # Calculate camera position in world coordinates
    if is_left_camera:
        camera_x = -baseline_mm / 2
    else:
        camera_x = baseline_mm / 2
    
    # Vector from camera to pupil in world coordinates
    dx = x_3d - camera_x
    dz = z_3d
    
    # Calculate actual viewing angle to this pupil position
    rotation_angle = np.arctan2(dx, dz)
    
    # Rotation matrix around Y axis
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    
    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    # Transform normal from camera to world coordinates
    normal_world = rotation_matrix @ normal_camera
    
    return normal_world


def calculate_pupil_radius_mm(ellipse, pupil_3d, is_left_camera, baseline_mm=36.442, 
                               fov_degrees=120, image_width=648):
    """
    Calculate physical pupil radius from ellipse major axis
    
    Parameters:
    - ellipse: ((cx, cy), (w, h), angle) ellipse parameters
    - pupil_3d: (x, y, z) 3D position of pupil in world coordinates
    - is_left_camera: True if left camera, False if right
    - baseline_mm: distance between cameras
    - fov_degrees: camera field of view
    - image_width: image width in pixels
    
    Returns:
    - pupil_radius_mm: physical radius in millimeters
    """
    x_3d, y_3d, z_3d = pupil_3d
    
    # Get major axis length in pixels
    (cx, cy), (w, h), angle = ellipse
    major_axis_px = max(w, h) / 2  # radius in pixels
    
    # Calculate camera position
    if is_left_camera:
        camera_x = -baseline_mm / 2
    else:
        camera_x = baseline_mm / 2
    
    # Distance from camera to pupil
    dx = x_3d - camera_x
    dy = y_3d
    dz = z_3d
    distance_to_pupil = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Calculate pixels per mm at this distance
    fov_rad = np.deg2rad(fov_degrees)
    sensor_width_mm = 2 * distance_to_pupil * np.tan(fov_rad / 2)
    pixels_per_mm = image_width / sensor_width_mm
    
    # Convert major axis from pixels to mm
    pupil_radius_mm = major_axis_px / pixels_per_mm
    
    return pupil_radius_mm


def reconstruct_3d_circle(ellipse_left, ellipse_right, pupil_3d, 
                          baseline_mm=36.442, fov_degrees=120, 
                          convergence_angle_degrees=36, image_width=648, image_height=486):
    """
    Reconstruct 3D circle (pupil) from two 2D ellipses using orthonormal vectors
    """
    if ellipse_left is None or ellipse_right is None or pupil_3d is None:
        return None
    
    x_3d, y_3d, z_3d = pupil_3d
    
    # Get normal vectors in camera coordinates
    left_n1_cam, left_n2_cam = ellipse_to_orthonormal_vectors(ellipse_left)
    right_n1_cam, right_n2_cam = ellipse_to_orthonormal_vectors(ellipse_right)
    
    if left_n1_cam is None or right_n1_cam is None:
        return None
    
    # Transform to world coordinates using actual 3D pupil position
    left_n1_world = camera_normal_to_world(left_n1_cam, pupil_3d, is_left_camera=True, baseline_mm=baseline_mm)
    left_n2_world = camera_normal_to_world(left_n2_cam, pupil_3d, is_left_camera=True, baseline_mm=baseline_mm)
    right_n1_world = camera_normal_to_world(right_n1_cam, pupil_3d, is_left_camera=False, baseline_mm=baseline_mm)
    right_n2_world = camera_normal_to_world(right_n2_cam, pupil_3d, is_left_camera=False, baseline_mm=baseline_mm)
    
    # Calculate pupil radius from both cameras and average
    radius_left = calculate_pupil_radius_mm(ellipse_left, pupil_3d, is_left_camera=True, 
                                           baseline_mm=baseline_mm, fov_degrees=fov_degrees, 
                                           image_width=image_width)
    radius_right = calculate_pupil_radius_mm(ellipse_right, pupil_3d, is_left_camera=False,
                                            baseline_mm=baseline_mm, fov_degrees=fov_degrees,
                                            image_width=image_width)
    pupil_radius_mm = (radius_left + radius_right) / 2
    
    return {
        'center': np.array([x_3d, y_3d, z_3d]),
        'left_normals': (left_n1_world, left_n2_world),
        'right_normals': (right_n1_world, right_n2_world),
        'radius_mm': pupil_radius_mm,
        'radius_left_mm': radius_left,
        'radius_right_mm': radius_right,
    }


def triangulate_converging_cameras(pupil_left, pupil_right, baseline_mm=36.442, 
                                   fov_degrees=120, convergence_angle_degrees=36,
                                   image_width=648, image_height=486):
    """
    Triangulate 3D pupil position from two converging cameras
    
    Parameters:
    - pupil_left: (x, y) in left camera coordinates
    - pupil_right: (x, y) in right camera coordinates  
    - baseline_mm: Distance between cameras (36.442mm)
    - fov_degrees: Horizontal field of view (120 degrees)
    - convergence_angle_degrees: Inward angle of each camera (54 degrees)
    - image_width: Width of image in pixels (648)
    
    Returns:
    - (x, y, z) in mm, where origin is midpoint between cameras,
      x is horizontal, y is vertical, z is depth (forward from camera plane)
    """
    if pupil_left is None or pupil_right is None:
        return None
    
    x_left, y_left = pupil_left
    x_right, y_right = pupil_right
    
    # Convert pixel coordinates to normalized coordinates [-1, 1]
    # (0, 0) at center of image
    x_left_norm = (x_left - image_width / 2) / (image_width / 2)
    x_right_norm = (x_right - image_width / 2) / (image_width / 2)
    
    # Calculate horizontal angle from camera optical axis for each view
    # FOV/2 gives angle from center to edge
    half_fov_rad = np.deg2rad(fov_degrees / 2)
    
    # Angle within camera's FOV
    angle_in_fov_left = x_left_norm * half_fov_rad
    angle_in_fov_right = x_right_norm * half_fov_rad
    
    # Convert to angles in world space
    # Left camera points inward (positive convergence angle)
    # Right camera points inward (negative convergence angle)
    convergence_rad = np.deg2rad(convergence_angle_degrees)
    
    # Angle of ray from left camera (measured from forward direction)
    # Left camera points to the RIGHT (positive angle)
    angle_left = convergence_rad + angle_in_fov_left
    
    # Angle of ray from right camera (measured from forward direction)
    # Right camera points to the LEFT (negative angle)
    angle_right = -convergence_rad + angle_in_fov_right
    
    # Camera positions in world coordinates
    # Left camera at (-baseline/2, 0, 0)
    # Right camera at (+baseline/2, 0, 0)
    cam_left_x = -baseline_mm / 2
    cam_right_x = baseline_mm / 2
    
    # Ray directions from each camera
    # Left camera ray: starts at (cam_left_x, 0) and goes at angle_left
    # Right camera ray: starts at (cam_right_x, 0) and goes at angle_right
    
    # Parametric form:
    # Left ray: (x, z) = (cam_left_x, 0) + t_left * (sin(angle_left), cos(angle_left))
    # Right ray: (x, z) = (cam_right_x, 0) + t_right * (sin(angle_right), cos(angle_right))
    
    # At intersection:
    # cam_left_x + t_left * sin(angle_left) = cam_right_x + t_right * sin(angle_right)
    # t_left * cos(angle_left) = t_right * cos(angle_right)
    
    # Solve for intersection using proper ray equations
    # Left ray: P_left = (cam_left_x, 0) + t_left * (sin(angle_left), cos(angle_left))
    # Right ray: P_right = (cam_right_x, 0) + t_right * (sin(angle_right), cos(angle_right))
    
    sin_left = np.sin(angle_left)
    cos_left = np.cos(angle_left)
    sin_right = np.sin(angle_right)
    cos_right = np.cos(angle_right)
    
    # At intersection: cam_left_x + t_left * sin_left = cam_right_x + t_right * sin_right
    #                  t_left * cos_left = t_right * cos_right
    
    # Solve system of equations
    # Using Cramer's rule or substitution
    denominator = sin_left * cos_right - sin_right * cos_left
    
    if abs(denominator) < 1e-6:
        return None  # Rays are parallel or nearly parallel
    
    # Solve for t_left and t_right
    t_left = ((cam_right_x - cam_left_x) * cos_right) / denominator
    t_right = ((cam_right_x - cam_left_x) * cos_left) / denominator
    
    # Calculate 3D position using left ray
    x_3d = cam_left_x + t_left * sin_left
    z_3d = t_left * cos_left
    
    # For y coordinate, average the y positions (assuming cameras are horizontally aligned)
    # Convert y pixel to angle and then to mm at depth z_3d
    y_avg_px = (y_left + y_right) / 2
    
    # Normalize Y coordinate: center at 0, flip so positive is UP
    # Pixel Y increases downward, but 3D Y should increase upward
    y_avg_norm = -(y_avg_px - image_height / 2) / (image_height / 2)
    
    # Calculate vertical FOV (assuming square pixels)
    vertical_fov_rad = half_fov_rad * (image_height / image_width)
    y_angle = y_avg_norm * vertical_fov_rad
    y_3d = z_3d * np.tan(y_angle)
    
    return (x_3d, y_3d, z_3d)


def main():
    video_path = select_video_file()
    
    # Stereo camera configuration
    TOP_IS_LEFT = True  # Set to False if top camera is on the right
    BASELINE_MM = 36.442
    FOV_DEGREES = 120
    CONVERGENCE_ANGLE = 36  # Angle from forward direction (90° - 54° from horizontal)
    IMAGE_WIDTH = 648
    
    # Create two pupil detector instances for stereo
    tracker_top = PupilDetector(window_name="(Top)")
    tracker_bottom = PupilDetector(window_name="(Bottom)")
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Split frame into top and bottom halves (stereo streams)
        height = frame.shape[0]
        frame_top = frame[:height//2, :].copy()
        frame_bottom = frame[height//2:, :].copy()
        
        # Track both streams
        result_top = tracker_top.track_frame(frame_top, frame_idx)
        result_bottom = tracker_bottom.track_frame(frame_bottom, frame_idx)
        
        # Triangulate 3D pupil position
        pupil_3d = None
        circle_3d = None
        if result_top is not None and result_bottom is not None:
            # Assign left/right based on configuration
            if TOP_IS_LEFT:
                pupil_left = (result_top['cx'], result_top['cy'])
                pupil_right = (result_bottom['cx'], result_bottom['cy'])
            else:
                pupil_left = (result_bottom['cx'], result_bottom['cy'])
                pupil_right = (result_top['cx'], result_top['cy'])
            
            # Triangulate 3D position
            pupil_3d = triangulate_converging_cameras(
                pupil_left, pupil_right,
                baseline_mm=BASELINE_MM,
                fov_degrees=FOV_DEGREES,
                convergence_angle_degrees=CONVERGENCE_ANGLE,
                image_width=IMAGE_WIDTH,
                image_height=frame_top.shape[0]
            )
            
            if pupil_3d is not None:
                x_mm, y_mm, z_mm = pupil_3d
                
                # Only use valid depth (positive Z, reasonable range)
                if z_mm > 5.0 and z_mm < 100.0:
                    
                    # Reconstruct 3D pupil circle from ellipses
                    # Get ellipse data from results
                    if TOP_IS_LEFT:
                        ellipse_left_data = result_top['best_ellipse_opencv']
                        ellipse_right_data = result_bottom['best_ellipse_opencv']
                    else:
                        ellipse_left_data = result_bottom['best_ellipse_opencv']
                        ellipse_right_data = result_top['best_ellipse_opencv']
                    
                    if ellipse_left_data is not None and ellipse_right_data is not None:
                        # Extract ellipse parameters (without crop offsets for now)
                        ellipse_left, _, _ = ellipse_left_data
                        ellipse_right, _, _ = ellipse_right_data
                        
                        # Reconstruct 3D circle
                        circle_3d = reconstruct_3d_circle(
                            ellipse_left, ellipse_right, pupil_3d,
                            baseline_mm=BASELINE_MM,
                            fov_degrees=FOV_DEGREES,
                            convergence_angle_degrees=CONVERGENCE_ANGLE,
                            image_width=IMAGE_WIDTH,
                            image_height=frame_top.shape[0]
                        )
                        
                        if circle_3d is not None:
                            pupil_radius_mm = circle_3d['radius_mm']
                            radius_left_mm = circle_3d['radius_left_mm']
                            radius_right_mm = circle_3d['radius_right_mm']
                            print(f"  Pupil radius: Left={radius_left_mm:.2f}mm, Right={radius_right_mm:.2f}mm, Avg={pupil_radius_mm:.2f}mm, Diff={radius_right_mm - radius_left_mm:.2f}mm")
                        
                    # Calculate sphere radius in pixels for each camera view
                    # The sphere center is at the eyeball center, pupil is on the surface
                    # We need to calculate the 2D pixel distance from sphere center to pupil
                    
                    EYE_RADIUS_MM = 13.0  # Real-world eye radius
                    half_fov_rad = np.deg2rad(FOV_DEGREES / 2)
                    convergence_rad = np.deg2rad(CONVERGENCE_ANGLE)
                    
                    # For each camera, calculate the distance from camera to the 3D pupil position
                    # Left camera is at (-baseline/2, 0, 0), right camera at (+baseline/2, 0, 0)
                    # Both cameras point inward at convergence_angle
                    
                    # Calculate distance from each camera to the pupil
                    if TOP_IS_LEFT:
                        # Top camera is left: at (-baseline/2, 0, 0)
                        camera_left_x = -BASELINE_MM / 2
                        camera_right_x = BASELINE_MM / 2
                    else:
                        # Top camera is right: at (+baseline/2, 0, 0)
                        camera_left_x = BASELINE_MM / 2
                        camera_right_x = -BASELINE_MM / 2
                    
                    # Distance from left camera to pupil (in camera's rotated coordinate system)
                    # Transform pupil position to left camera's coordinate system
                    pupil_rel_left_x = x_mm - camera_left_x
                    pupil_rel_left_z = z_mm
                    # Rotate by convergence angle to get distance along camera's optical axis
                    dist_left_z = pupil_rel_left_x * np.sin(convergence_rad) + pupil_rel_left_z * np.cos(convergence_rad)
                    
                    # Distance from right camera to pupil
                    pupil_rel_right_x = x_mm - camera_right_x
                    pupil_rel_right_z = z_mm
                    dist_right_z = -pupil_rel_right_x * np.sin(convergence_rad) + pupil_rel_right_z * np.cos(convergence_rad)
                    
                    # Calculate pixels per mm at each camera's viewing distance
                    physical_width_left = dist_left_z * np.tan(half_fov_rad) * 2
                    pixels_per_mm_left = IMAGE_WIDTH / physical_width_left
                    sphere_radius_left = int(EYE_RADIUS_MM * pixels_per_mm_left)
                    
                    physical_width_right = dist_right_z * np.tan(half_fov_rad) * 2
                    pixels_per_mm_right = IMAGE_WIDTH / physical_width_right
                    sphere_radius_right = int(EYE_RADIUS_MM * pixels_per_mm_right)
                    
                    # Clamp to reasonable range
                    sphere_radius_left = max(20, min(sphere_radius_left, 300))
                    sphere_radius_right = max(20, min(sphere_radius_right, 300))
                    
                    # Update each tracker with its own sphere radius
                    if TOP_IS_LEFT:
                        tracker_top.max_observed_distance = sphere_radius_left
                        tracker_bottom.max_observed_distance = sphere_radius_right
                    else:
                        tracker_top.max_observed_distance = sphere_radius_right
                        tracker_bottom.max_observed_distance = sphere_radius_left
                    
                    # print(f"  Sphere radius: Left={sphere_radius_left}px, Right={sphere_radius_right}px")
                else:
                    print(f"Frame {frame_idx}: Invalid depth {z_mm:.2f}mm (skipping)")
        
        # Display results on full frame
        display_results(frame, result_top, result_bottom, frame_idx,
                       sphere_radius_top=tracker_top.max_observed_distance,
                       sphere_radius_bottom=tracker_bottom.max_observed_distance,
                       pupil_3d=pupil_3d,
                       circle_3d=circle_3d)
        
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

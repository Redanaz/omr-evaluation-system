import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional

# ---------- Mobile Camera Optimized Configuration ----------
SUBJECTS = {
    "PYTHON": list(range(1, 21)),          # Questions 1-20
    "DATA ANALYSIS": list(range(21, 41)),  # Questions 21-40
    "MySQL": list(range(41, 61)),          # Questions 41-60
    "POWER BI": list(range(61, 81)),       # Questions 61-80
    "Adv STATS": list(range(81, 101))      # Questions 81-100
}

# Mobile camera optimized parameters
BUBBLE_MIN_AREA = 100      # Smaller for mobile images
BUBBLE_MAX_AREA = 1200     # Larger range for mobile images
FILL_THRESHOLD = 0.40      # Lower threshold for mobile camera detection
MIN_CONTOUR_AREA_RATIO = 0.1  # Minimum ratio of image area for sheet detection

# ---------- Mobile Camera Enhancement Functions ----------

def enhance_mobile_image(img):
    """Enhanced preprocessing specifically for mobile camera images"""
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Resize if image is too large (common with mobile cameras)
    height, width = gray.shape
    if width > 2000 or height > 2000:
        scale_factor = min(2000/width, 2000/height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Apply adaptive histogram equalization for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur to reduce mobile camera noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Sharpen to enhance bubble edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    return sharpened

def detect_sheet_mobile(image):
    """Improved sheet detection for mobile camera images with perspective correction"""
    gray = enhance_mobile_image(image)
    
    # Multiple edge detection approaches for mobile images
    edges1 = cv2.Canny(gray, 30, 100, apertureSize=3)
    edges2 = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges3 = cv2.Canny(gray, 80, 200, apertureSize=3)
    
    # Combine edge detection results
    edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    
    # Morphological operations to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Look for the largest rectangular contour (the OMR sheet)
    sheet_contour = None
    image_area = image.shape[0] * image.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Sheet should be at least 10% of image area
        if area < image_area * MIN_CONTOUR_AREA_RATIO:
            continue
            
        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)  # More flexible approximation
        
        # Look for quadrilateral (4 corners)
        if len(approx) >= 4:
            # If more than 4 points, take the 4 corner points
            if len(approx) > 4:
                # Sort by distance from corners and take 4 most corner-like points
                hull = cv2.convexHull(approx)
                if len(hull) >= 4:
                    approx = hull[:4]
            
            sheet_contour = approx
            break
    
    return sheet_contour

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference method
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]   # Bottom-right (largest sum)
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right (smallest difference)
    rect[3] = pts[np.argmax(diff)]  # Bottom-left (largest difference)
    
    return rect

def preprocess_mobile_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Enhanced preprocessing for mobile camera images"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    original = img.copy()
    
    # Detect sheet boundary
    sheet_contour = detect_sheet_mobile(img)
    
    if sheet_contour is None:
        # If automatic detection fails, return enhanced original
        enhanced = enhance_mobile_image(original)
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        print("Warning: Could not detect sheet boundaries, using whole image")
        return enhanced_color, None
    
    # Extract and order corner points
    pts = sheet_contour.reshape(-1, 2).astype(np.float32)
    if len(pts) == 4:
        rect = order_points(pts)
    else:
        # If we have more than 4 points, find the best 4 corners
        # Use convex hull and find extreme points
        hull = cv2.convexHull(pts)
        rect = order_points(hull.reshape(-1, 2))
    
    (tl, tr, br, bl) = rect
    
    # Calculate dimensions for perspective correction
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    # Ensure minimum dimensions
    maxWidth = max(maxWidth, 800)
    maxHeight = max(maxHeight, 1000)
    
    # Define destination points for perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Apply perspective transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
    
    # Additional enhancement after perspective correction
    enhanced_warp = enhance_mobile_image(warp)
    final_warp = cv2.cvtColor(enhanced_warp, cv2.COLOR_GRAY2BGR)
    
    return final_warp, sheet_contour

def detect_bubbles_mobile(thresh_img: np.ndarray) -> List[Tuple]:
    """Enhanced bubble detection for mobile camera images"""
    # Apply additional morphological operations for mobile images
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if BUBBLE_MIN_AREA <= area <= BUBBLE_MAX_AREA:
            # Enhanced shape analysis for mobile images
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # More flexible aspect ratio for mobile camera distortion
            if 0.6 <= aspect_ratio <= 1.4:
                # Enhanced circularity check
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # More lenient circularity for mobile images
                    if circularity > 0.3:
                        # Additional solidity check (filled ratio)
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = area / hull_area
                            if solidity > 0.7:  # Should be reasonably filled
                                bubbles.append((x, y, w, h, cnt, area))
    
    return bubbles

def detect_answers_mobile(warp: np.ndarray, num_questions: int = 100) -> Tuple[Dict, np.ndarray]:
    """Enhanced answer detection optimized for mobile camera images"""
    if warp is None:
        return {}, warp
    
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_mobile_image(gray)
    
    # Adaptive thresholding for mobile images with uneven lighting
    thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Combine thresholding methods
    thresh = cv2.bitwise_or(thresh1, thresh2)
    
    # Detect bubbles with mobile optimization
    bubbles = detect_bubbles_mobile(thresh)
    
    if len(bubbles) < num_questions * 2:  # At least 200 bubbles expected
        print(f"Warning: Only detected {len(bubbles)} bubbles for mobile image")
    
    # Group bubbles by rows with mobile-friendly tolerance
    def group_bubbles_by_rows_mobile(bubbles: List[Tuple], tolerance: int = 20) -> Dict[int, List]:
        if not bubbles:
            return {}
        
        bubbles_sorted = sorted(bubbles, key=lambda b: b[1])  # Sort by y-coordinate
        rows = {}
        current_row = 0
        current_y = bubbles_sorted[0][1]
        
        for bubble in bubbles_sorted:
            x, y, w, h, cnt, area = bubble
            
            if abs(y - current_y) > tolerance:
                current_row += 1
                current_y = y
            
            if current_row not in rows:
                rows[current_row] = []
            rows[current_row].append(bubble)
        
        return rows
    
    rows = group_bubbles_by_rows_mobile(bubbles)
    
    answers = {}
    overlay = warp.copy()
    
    question_num = 0
    for row_idx in sorted(rows.keys()):
        if question_num >= num_questions:
            break
            
        row_bubbles = rows[row_idx]
        row_bubbles = sorted(row_bubbles, key=lambda b: b[0])  # Sort by x-coordinate
        
        # Process bubbles in groups of 4 (A, B, C, D)
        for i in range(0, len(row_bubbles) - 3, 4):
            if question_num >= num_questions:
                break
                
            bubble_group = row_bubbles[i:i+4]
            
            # Enhanced fill detection for mobile images
            filled_options = []
            fill_ratios = []
            
            for option_idx, (x, y, w, h, cnt, area) in enumerate(bubble_group):
                # Create mask
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                # Count filled pixels
                filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                fill_ratio = filled_pixels / area
                fill_ratios.append(fill_ratio)
                
                if fill_ratio > FILL_THRESHOLD:
                    filled_options.append(option_idx)
            
            # Smart decision making for mobile images
            if len(filled_options) == 1:
                answers[question_num] = filled_options[0]
                # Mark as correctly detected (green)
                cv2.drawContours(overlay, [bubble_group[filled_options[0]][4]], -1, (0, 255, 0), 3)
            elif len(filled_options) > 1:
                # Multiple answers - choose the one with highest fill ratio
                best_option = max(filled_options, key=lambda x: fill_ratios[x])
                answers[question_num] = best_option
                # Mark as multiple but selected (yellow)
                cv2.drawContours(overlay, [bubble_group[best_option][4]], -1, (0, 255, 255), 3)
            else:
                # No clear answer - check if any bubble has reasonable fill ratio
                if max(fill_ratios) > FILL_THRESHOLD * 0.7:  # 70% of threshold
                    best_option = fill_ratios.index(max(fill_ratios))
                    answers[question_num] = best_option
                    # Mark as weak detection (orange)
                    cv2.drawContours(overlay, [bubble_group[best_option][4]], -1, (0, 165, 255), 2)
                else:
                    answers[question_num] = -2  # No answer
            
            # Mark all bubbles for visibility
            for option_idx, (x, y, w, h, cnt, area) in enumerate(bubble_group):
                if option_idx not in filled_options:
                    cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 1)  # Red for unfilled
            
            question_num += 1
    
    return answers, overlay

def calculate_subject_scores(answers: Dict, results: List) -> Dict[str, int]:
    """Calculate scores for each subject"""
    subject_scores = {subject: 0 for subject in SUBJECTS.keys()}
    
    for question_num, answer, correct_answer, is_correct in results:
        if is_correct:
            for subject, question_range in SUBJECTS.items():
                if question_num in question_range:
                    subject_scores[subject] += 1
                    break
    
    return subject_scores

def evaluate(image_path: str, answer_key: Dict[int, int], student_id: str = "Student") -> Tuple:
    """Main evaluation function optimized for mobile camera images"""
    try:
        # Mobile-optimized preprocessing
        warp, contour = preprocess_mobile_image(image_path)
        if warp is None:
            return None, None, "Sheet not detected - ensure good lighting and full sheet visibility", {}, student_id
        
        # Mobile-optimized answer detection
        answers, overlay = detect_answers_mobile(warp, num_questions=len(answer_key))
        
        if not answers:
            return overlay, [], "No answers detected - check image quality and bubble filling", {}, student_id
        
        # Evaluate answers
        score = 0
        results = []
        
        for question_num in range(1, len(answer_key) + 1):
            question_idx = question_num - 1
            student_answer = answers.get(question_idx, -2)
            correct_answer = answer_key.get(question_idx, 0)
            
            is_correct = False
            if student_answer >= 0 and student_answer == correct_answer:
                is_correct = True
                score += 1
            
            results.append((question_num, student_answer, correct_answer, is_correct))
        
        # Calculate subject scores
        subject_scores = calculate_subject_scores(answers, results)
        
        # Save results
        save_results(student_id, overlay, results, subject_scores, score)
        
        return overlay, results, score, subject_scores, student_id
        
    except Exception as e:
        print(f"Error evaluating mobile image {image_path}: {str(e)}")
        return None, None, f"Processing error: {str(e)}", {}, student_id

def save_results(student_id: str, overlay: np.ndarray, results: List, 
                subject_scores: Dict[str, int], score: int):
    """Save evaluation results"""
    os.makedirs("results", exist_ok=True)
    
    if overlay is not None:
        cv2.imwrite(f"results/{student_id}_overlay.png", overlay)
    
    result_data = {
        "student": student_id,
        "total_score": score,
        "percentage": (score / 100) * 100,
        "subject_scores": subject_scores,
        "detailed_results": []
    }
    
    for question_num, student_ans, correct_ans, is_correct in results:
        answer_text = "No Answer"
        if student_ans == -1:
            answer_text = "Multiple Answers"
        elif student_ans >= 0:
            answer_text = chr(65 + student_ans)
        
        result_data["detailed_results"].append({
            "question": question_num,
            "student_answer": answer_text,
            "correct_answer": chr(65 + correct_ans),
            "is_correct": is_correct,
            "subject": get_subject_for_question(question_num)
        })
    
    with open(f"results/{student_id}.json", "w") as f:
        json.dump(result_data, f, indent=2)

def get_subject_for_question(question_num: int) -> str:
    """Get subject name for a given question number"""
    for subject, question_range in SUBJECTS.items():
        if question_num in question_range:
            return subject
    return "Unknown"

def validate_answer_key(answer_key: Dict[int, int]) -> bool:
    """Validate answer key format"""
    if len(answer_key) != 100:
        print(f"Error: Answer key should have 100 questions, found {len(answer_key)}")
        return False
    
    for q_idx, answer in answer_key.items():
        if not isinstance(answer, int) or answer < 0 or answer > 3:
            print(f"Error: Invalid answer {answer} for question {q_idx + 1}")
            return False
    
    return True
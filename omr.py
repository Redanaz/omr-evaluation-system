import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional

# ---------- Configuration ----------
SUBJECTS = {
    "PYTHON": list(range(1, 21)),          # Questions 1-20
    "DATA ANALYSIS": list(range(21, 41)),  # Questions 21-40
    "MySQL": list(range(41, 61)),          # Questions 41-60
    "POWER BI": list(range(61, 81)),       # Questions 61-80
    "Adv STATS": list(range(81, 101))      # Questions 81-100
}

BUBBLE_MIN_AREA = 150
BUBBLE_MAX_AREA = 800
FILL_THRESHOLD = 0.45  # 45% filled to consider marked

# ---------- Utility Functions ----------

def order_points(pts):
    """Order points in the order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-left
    rect[2] = pts[np.argmax(s)]   # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def enhance_image(img):
    """Apply image enhancement techniques"""
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply slight gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred

def preprocess_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Preprocess image and detect OMR sheet boundaries"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance the image
    enhanced = enhance_image(gray)
    
    # Apply morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    # Edge detection with multiple thresholds
    edges1 = cv2.Canny(morphed, 50, 150)
    edges2 = cv2.Canny(morphed, 80, 200)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Dilate edges to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sheet_contour = None
    for cnt in contours:
        # Calculate contour area and filter by minimum size
        area = cv2.contourArea(cnt)
        if area < img.shape[0] * img.shape[1] * 0.3:  # At least 30% of image
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            sheet_contour = approx
            break

    if sheet_contour is None:
        # Return original image if sheet boundary not detected
        print("Warning: Sheet boundary not detected, using original image")
        return original, None

    pts = sheet_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate dimensions
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

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
    
    return warp, sheet_contour

def detect_bubbles(thresh_img: np.ndarray) -> List[Tuple]:
    """Detect circular bubbles in the image"""
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if BUBBLE_MIN_AREA <= area <= BUBBLE_MAX_AREA:
            # Check if contour is roughly circular
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # Filter by aspect ratio (should be close to 1 for circles)
            if 0.7 <= aspect_ratio <= 1.3:
                # Calculate circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Reasonably circular
                        bubbles.append((x, y, w, h, cnt, area))
    
    return bubbles

def group_bubbles_by_rows(bubbles: List[Tuple], tolerance: int = 15) -> Dict[int, List]:
    """Group bubbles by rows based on y-coordinate"""
    if not bubbles:
        return {}
    
    # Sort bubbles by y-coordinate
    bubbles_sorted = sorted(bubbles, key=lambda b: b[1])
    
    rows = {}
    current_row = 0
    current_y = bubbles_sorted[0][1]
    
    for bubble in bubbles_sorted:
        x, y, w, h, cnt, area = bubble
        
        # If y-coordinate is significantly different, start new row
        if abs(y - current_y) > tolerance:
            current_row += 1
            current_y = y
        
        if current_row not in rows:
            rows[current_row] = []
        rows[current_row].append(bubble)
    
    return rows

def detect_answers_improved(warp: np.ndarray, num_questions: int = 100) -> Tuple[Dict, np.ndarray]:
    """Improved answer detection with better bubble recognition"""
    if warp is None:
        return {}, warp
    
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    
    # Enhance image before thresholding
    enhanced = enhance_image(gray)
    
    # Apply threshold
    thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Detect bubbles
    bubbles = detect_bubbles(thresh)
    
    if len(bubbles) < num_questions * 4:  # Should have at least 400 bubbles
        print(f"Warning: Only detected {len(bubbles)} bubbles, expected {num_questions * 4}")
    
    # Group bubbles by rows
    rows = group_bubbles_by_rows(bubbles)
    
    answers = {}
    overlay = warp.copy()
    
    question_num = 0
    for row_idx in sorted(rows.keys()):
        row_bubbles = rows[row_idx]
        
        # Sort bubbles in row by x-coordinate
        row_bubbles = sorted(row_bubbles, key=lambda b: b[0])
        
        # Process bubbles in groups of 4 (A, B, C, D)
        for i in range(0, len(row_bubbles), 4):
            if question_num >= num_questions:
                break
                
            bubble_group = row_bubbles[i:i+4]
            if len(bubble_group) != 4:
                continue
            
            # Check which bubbles are filled
            filled_options = []
            max_filled_pixels = 0
            
            for option_idx, (x, y, w, h, cnt, area) in enumerate(bubble_group):
                # Create mask for this bubble
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                # Count filled pixels
                filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                fill_ratio = filled_pixels / area
                
                max_filled_pixels = max(max_filled_pixels, filled_pixels)
                
                if fill_ratio > FILL_THRESHOLD:
                    filled_options.append(option_idx)
                    # Mark as filled (green)
                    cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
                else:
                    # Mark as unfilled (red outline)
                    cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 1)
            
            # Determine answer
            if len(filled_options) == 1:
                answers[question_num] = filled_options[0]  # Single correct answer
            elif len(filled_options) > 1:
                answers[question_num] = -1  # Multiple answers marked
                # Highlight multiple answers in blue
                for option_idx in filled_options:
                    cv2.drawContours(overlay, [bubble_group[option_idx][4]], -1, (255, 0, 0), 2)
            else:
                answers[question_num] = -2  # No answer marked
            
            question_num += 1
    
    return answers, overlay

def calculate_subject_scores(answers: Dict, results: List) -> Dict[str, int]:
    """Calculate scores for each subject"""
    subject_scores = {subject: 0 for subject in SUBJECTS.keys()}
    
    for question_num, answer, correct_answer, is_correct in results:
        if is_correct:
            # Find which subject this question belongs to
            for subject, question_range in SUBJECTS.items():
                if question_num in question_range:
                    subject_scores[subject] += 1
                    break
    
    return subject_scores

def evaluate(image_path: str, answer_key: Dict[int, int], student_id: str = "Student") -> Tuple:
    """Main evaluation function"""
    try:
        # Preprocess image
        warp, contour = preprocess_image(image_path)
        if warp is None:
            return None, None, "Sheet not detected", {}, student_id
        
        # Detect answers
        answers, overlay = detect_answers_improved(warp, num_questions=len(answer_key))
        
        if not answers:
            return overlay, [], "No answers detected", {}, student_id
        
        # Evaluate answers
        score = 0
        results = []
        
        for question_num in range(1, len(answer_key) + 1):
            question_idx = question_num - 1  # Convert to 0-based indexing
            student_answer = answers.get(question_idx, -2)  # -2 means no answer
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
        print(f"Error evaluating {image_path}: {str(e)}")
        return None, None, f"Error: {str(e)}", {}, student_id

def save_results(student_id: str, overlay: np.ndarray, results: List, 
                subject_scores: Dict[str, int], score: int):
    """Save evaluation results"""
    os.makedirs("results", exist_ok=True)
    
    # Save overlay image
    if overlay is not None:
        cv2.imwrite(f"results/{student_id}_overlay.png", overlay)
    
    # Save detailed results
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
            answer_text = chr(65 + student_ans)  # Convert to A, B, C, D
        
        result_data["detailed_results"].append({
            "question": question_num,
            "student_answer": answer_text,
            "correct_answer": chr(65 + correct_ans),
            "is_correct": is_correct,
            "subject": get_subject_for_question(question_num)
        })
    
    # Save as JSON
    with open(f"results/{student_id}.json", "w") as f:
        json.dump(result_data, f, indent=2)

def get_subject_for_question(question_num: int) -> str:
    """Get subject name for a given question number"""
    for subject, question_range in SUBJECTS.items():
        if question_num in question_range:
            return subject
    return "Unknown"

# ---------- Answer Key Validation ----------
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
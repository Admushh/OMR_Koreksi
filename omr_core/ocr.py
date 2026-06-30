import cv2
import numpy as np
import os
import sys

# Disable oneDNN/MKLDNN execution in paddle to bypass Windows CPU PIR bug
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        # Import paddleocr locally to prevent import delay during startup
        from paddleocr import PaddleOCR 
        # Initialize PaddleOCR engine for CPU with orientation classifiers disabled
        _ocr_engine = PaddleOCR(
            lang='en',
            use_doc_orientation_classify=False,
            use_textline_orientation=False
        )
    return _ocr_engine


def remove_grid_lines(crop_img):
    """
    Pad the crop with white space for OCR engine.
    (Grid line removal is skipped to keep the printed lines intact).
    """
    if len(crop_img.shape) == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img.copy()
    padded = cv2.copyMakeBorder(gray, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return padded


def map_lookalike_letters(text: str) -> str:
    """
    Map common handwritten digit misrecognitions to their alphabetical equivalents.
    For example: '1' -> 'I', '5' -> 'S', '0' -> 'O', etc.
    """
    mapping = {
        '1': 'I',
        '2': 'Z',
        '3': 'E',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B',
        '9': 'G',
        '0': 'O'
    }
    return "".join(mapping.get(char, char) for char in text)


def map_lookalike_digits(text: str) -> str:
    """
    Map common handwritten letter misrecognitions to their numeric equivalents.
    For example: 'i'/'I'/'l' -> '1', 'g'/'q' -> '9', 'S'/'s' -> '5', etc.
    """
    mapping = {
        'i': '1', 'I': '1', 'l': '1', 'L': '1', '|': '1',
        'z': '2', 'Z': '2',
        'e': '3', 'E': '3',
        'A': '4',
        's': '5', 'S': '5',
        'b': '6', 'G': '6',
        't': '7', 'T': '7',
        'B': '8',
        'g': '9', 'q': '9',
        'O': '0', 'o': '0', 'D': '0'
    }
    return "".join(mapping.get(char, char) for char in text)


def get_name_id_y_coords(warped_gray):
    """
    Locate the exact top, middle, and bottom boundaries of the Name/ID boxes
    dynamically using row averages (projection profile) on the grayscale image.
    Returns (y_top, y_mid, y_bot).
    """
    h_img, w_img = warped_gray.shape[:2]
    # Restrict search area to X in [245:780] and Y in [160:280]
    y_start, y_end = max(0, 160), min(h_img, 280)
    x_start, x_end = max(0, 245), min(w_img, 780)
    
    roi = warped_gray[y_start:y_end, x_start:x_end]
    row_means = np.mean(roi, axis=1)
    
    # Find local minima (where the black horizontal border lines lie)
    minima = []
    for y in range(2, len(row_means) - 2):
        val = row_means[y]
        if val < row_means[y-1] and val < row_means[y-2] and val < row_means[y+1] and val < row_means[y+2]:
            if val < 220:  # Grid lines are significantly darker than paper background
                actual_y = y + y_start
                if not minima or actual_y - minima[-1] > 10:
                    minima.append(actual_y)
                    
    # We expect 3 horizontal borders (Name top, middle divider, ID bottom)
    if len(minima) >= 3:
        # Search for a sequence of 3 peaks with reasonable spacing (each row is ~30px tall)
        for i in range(len(minima) - 2):
            p1, p2, p3 = minima[i], minima[i+1], minima[i+2]
            if 20 <= (p2 - p1) <= 45 and 20 <= (p3 - p2) <= 45:
                return p1, p2, p3
                
    # Fallback to defaults if detection fails
    return 189, 219, 249


def get_grid_x_bounds(warped_gray, y_top, y_bot):
    """
    Locate the exact starting X coordinate of the cell grid (first vertical divider)
    and the ending X coordinates for Name and ID rows using a multi-line comb search.
    """
    roi = warped_gray[y_top:y_bot, :]
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_sums = np.sum(thresh, axis=0)
    
    best_x_start = 220
    best_cell_width = 42.7
    max_score = 0
    
    # Search grid start X in range [215:230] and cell width in [41.5:43.5]
    for x_start in range(215, 230):
        for w_val in range(415, 436, 1):
            cell_width = w_val / 10.0
            score = 0
            for i in range(14):  # 14 vertical line dividers
                idx = int(round(x_start + i * cell_width))
                if idx < len(col_sums):
                    score += np.max(col_sums[idx - 1 : idx + 2])
            if score > max_score:
                max_score = score
                best_x_start = x_start
                best_cell_width = cell_width
                
    x_end_name = int(round(best_x_start + 13 * best_cell_width))
    x_end_id = int(round(best_x_start + 10 * best_cell_width))
    
    return best_x_start, x_end_name, x_end_id


def _extract_text_from_result(ocr_res) -> str:
    if not ocr_res:
        return ""
    
    first_item = ocr_res[0]
    if hasattr(first_item, "get"):
        texts = first_item.get("rec_texts", [])
        return " ".join(str(t) for t in texts)
        
    if isinstance(ocr_res, list):
        if isinstance(first_item, tuple) and len(first_item) == 2 and isinstance(first_item[0], str):
            return first_item[0]
            
        if isinstance(first_item, list):
            if len(first_item) > 0 and isinstance(first_item[0], tuple) and len(first_item[0]) == 2 and isinstance(first_item[0][0], str):
                return first_item[0][0]
                
            texts = []
            for line in first_item:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                        texts.append(str(text_info[0]))
            return " ".join(texts)
            
    return ""


def extract_name_and_id(warped_gray):
    """
    Extract Name and ID (Nomor Induk) text from the warped grayscale sheet image.
    Determines coordinates dynamically to handle vertical and horizontal offsets.
    """
    # Dynamically locate horizontal line coordinates
    y_top, y_mid, y_bot = get_name_id_y_coords(warped_gray)

    # Dynamically locate the vertical divider lines of the cell grid
    x_start, x_end_name, x_end_id = get_grid_x_bounds(warped_gray, y_top, y_bot)

    # Crop Name and ID ROIs using precise dynamic coordinates matching the grid lines
    name_crop = warped_gray[y_top : y_mid + 1, x_start : x_end_name + 1]
    id_crop = warped_gray[y_mid : y_bot + 1, x_start : x_end_id + 1]

    # Pre-process crops by dynamically removing grid borders and padding
    name_cleaned = remove_grid_lines(name_crop)
    id_cleaned = remove_grid_lines(id_crop)

    # Save cleaned images to the root directory for inspection and user/frontend visibility
    cv2.imwrite("scratch_ocr_cleaned_name.png", name_cleaned)
    cv2.imwrite("scratch_ocr_cleaned_id.png", id_cleaned)

    # Convert to BGR format (expected by PaddleOCR)
    name_bgr = cv2.cvtColor(name_cleaned, cv2.COLOR_GRAY2BGR)
    id_bgr = cv2.cvtColor(id_cleaned, cv2.COLOR_GRAY2BGR)

    # Perform OCR
    engine = get_ocr_engine()
    
    # 1. OCR Name
    name_text = ""
    try:
        name_res = engine.ocr(name_bgr)
        raw_name = _extract_text_from_result(name_res).strip().upper()
        # Map lookalike numbers to letters (e.g. '1' -> 'I', '5' -> 'S')
        mapped_name = map_lookalike_letters(raw_name)
        # Clean: keep only letters and spaces
        name_text = "".join([c for c in mapped_name if c.isalpha() or c.isspace()]).strip()
        # Normalize spaces (collapse multiple spaces to single space)
        name_text = " ".join(name_text.split())
    except Exception as e:
        print(f"[OCR] Error scanning name: {e}")

    # 2. OCR ID
    id_text = ""
    try:
        id_res = engine.ocr(id_bgr)
        raw_id = _extract_text_from_result(id_res).strip()
        # Map lookalike characters to digits (e.g. 'i'/'I' -> '1', 'g' -> '9')
        mapped_id = map_lookalike_digits(raw_id)
        # Clean: keep only digits
        id_text = "".join([c for c in mapped_id if c.isdigit()])
    except Exception as e:
        print(f"[OCR] Error scanning ID: {e}")

    return name_text, id_text

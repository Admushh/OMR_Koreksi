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
    Remove horizontal and vertical grid lines from Name/ID boxes cleanly.
    Identifies grid cells using contour detection, erases their boundaries,
    filters out noise dots, and pads the crop with white space for high-accuracy OCR.
    """
    if len(crop_img.shape) == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img.copy()
        
    h, w = gray.shape
    
    # 1. Binarize (invert so lines/text are white, paper is black)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Find contours of the cells in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that match the cell size
    cell_boxes = []
    for c in contours:
        x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        aspect = w_box / float(h_box) if h_box > 0 else 0
        
        # Grid cell dimensions: width around 25-55, height around 20-40
        if 25 < w_box < 55 and 20 < h_box < 40 and 0.8 < aspect < 2.0:
            cell_boxes.append((x_box, y_box, w_box, h_box))
            
    # 3. Erase the borders of all detected cells (draw black rectangles on binarized image)
    cleaned = thresh.copy()
    for (x_box, y_box, w_box, h_box) in cell_boxes:
        cv2.rectangle(cleaned, (x_box, y_box), (x_box + w_box, y_box + h_box), 0, 3)
        
    # Also erase top/bottom borders using horizontal morphology to catch any left-over edge lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    kernel_h_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    detect_horizontal = cv2.dilate(detect_horizontal, kernel_h_dilate, iterations=1)
    
    cleaned = cv2.subtract(cleaned, detect_horizontal)
    
    # 4. Remove small noise components (like leftover dots from grid intersections)
    # Any component with an area less than 12 pixels is erased.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 12:
            cleaned[labels == i] = 0
            
    # 5. Dilate slightly to heal characters
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)
    
    # 6. Invert back to standard black text on white paper
    cleaned_inverted = cv2.bitwise_not(cleaned)
    
    # 7. Pad with white pixels (15px top/bottom, 20px left/right) to give OCR breathing room
    padded = cv2.copyMakeBorder(cleaned_inverted, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return padded


def extract_name_and_id(warped_gray):
    """
    Extract Name and ID (Nomor Induk) text from the warped grayscale sheet image.
    
    Name ROI: Y[185:225], X[245:780]
    Nomor Induk ROI: Y[215:255], X[245:660]
    """
    # Crop ROIs
    name_crop = warped_gray[185:225, 245:780]
    id_crop = warped_gray[215:255, 245:660]

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
        if name_res and name_res[0]:
            name_text = " ".join(name_res[0].get('rec_texts', [])).strip().upper()
    except Exception as e:
        print(f"[OCR] Error scanning name: {e}")

    # 2. OCR ID
    id_text = ""
    try:
        id_res = engine.ocr(id_bgr)
        if id_res and id_res[0]:
            raw_id = "".join(id_res[0].get('rec_texts', [])).strip()
            # Clean: keep only digits
            id_text = "".join([c for c in raw_id if c.isdigit()])
    except Exception as e:
        print(f"[OCR] Error scanning ID: {e}")

    return name_text, id_text

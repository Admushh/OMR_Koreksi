"""
preprocess.py — Dual-mode OMR image preprocessing (v3-robust)
==============================================================

Two separate pipelines optimized for different detection tasks:
  1. preprocess_for_markers()  → binary image for corner marker detection
  2. preprocess_for_answers()  → enhanced grayscale for bubble scoring
"""

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. MARKER DETECTION PREPROCESSING
#    Goal: clean binary image where corner markers are prominent
# ═══════════════════════════════════════════════════════════════

def preprocess_for_markers(image):
    """
    Preprocess for corner marker detection.
    
    Strategy:
    - CLAHE to normalize uneven lighting (shadows, hand noise)
    - Otsu threshold for robust binarization
    - Morphological cleaning WITHOUT masking corners
    - NO line removal (irrelevant for markers, risks damaging them)
    
    Returns: binary image (white = foreground/ink)
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. CLAHE — normalize lighting across the image
    #    clipLimit=2.0 handles shadows (like hand in sample 2.png)
    #    better than 1.2 which is too gentle
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Gaussian Blur — reduce paper texture noise
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 4. Otsu Threshold — auto-picks optimal threshold
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 5. Morphological Open — remove small noise specks
    #    Kernel 3x3 removes dots without eroding marker shapes
    kernel_noise = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise)

    # NOTE: NO corner masking, NO line removal here
    # Markers need to be fully intact for detection
    return thresh


# ═══════════════════════════════════════════════════════════════
# 2. ANSWER/BUBBLE DETECTION PREPROCESSING  
#    Goal: enhanced grayscale where filled bubbles are clearly
#          darker than empty bubbles
# ═══════════════════════════════════════════════════════════════

def preprocess_for_answers(warped_gray):
    """
    Preprocess a warped grayscale image for bubble detection.
    
    Input:  grayscale image (already perspective-corrected to 1000x1414)
    Output: enhanced grayscale image (NOT binary)
    
    Why grayscale instead of binary?
    - Binary loses intensity information
    - Filled bubble (dark pencil) vs empty bubble (thin border) contrast
      is best measured by mean pixel intensity, not pixel count
    - Otsu on warped image often over-thresholds, making empty bubble
      borders look "filled"
    """
    if len(warped_gray.shape) == 3:
        warped_gray = cv2.cvtColor(warped_gray, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE — enhanced local contrast
    #    Higher tileGrid for more localized normalization on warped canvas
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
    enhanced = clahe.apply(warped_gray)

    # 2. Light blur to smooth pencil texture  
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced


# ═══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════

def preprocess_image(image):
    """
    Legacy wrapper — calls preprocess_for_markers().
    Kept for backward compatibility with main.py and existing scripts.
    """
    return preprocess_for_markers(image)
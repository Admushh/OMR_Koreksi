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
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. CLAHE — normalize lighting across the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Gaussian Blur — reduce paper texture noise
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

# 4. Adaptive Thresholding — kebal terhadap bayangan (shadow)
    thresh = cv2.adaptiveThreshold(
        blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        75,  
        15   
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
    Menggunakan teknik Background Subtraction untuk menghilangkan bayangan ekstrem.
    """
    if len(warped_gray.shape) == 3:
        warped_gray = cv2.cvtColor(warped_gray, cv2.COLOR_BGR2GRAY)

    # 1. ESTIMASI BACKGROUND (Menangkap Pola Bayangan)
    # Kita pakai kernel lingkaran berukuran 35x35 (harus lebih besar dari ukuran 1 buletan LJK)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    
    # DILATE: Algoritma ini akan "menelan" semua objek gelap (tinta/pensil),
    # sehingga yang tersisa HANYA warna dasar kertas beserta gradasi bayangannya.
    background = cv2.morphologyEx(warped_gray, cv2.MORPH_DILATE, kernel)
    
    # Blur sedikit agar transisi bayangannya mulus
    background = cv2.medianBlur(background, 21)

    # 2. HAPUS BAYANGAN (Subtraksi)
    # Gambar background dikurangi gambar asli. 
    # Area bayangan (gelap - gelap) jadi 0 (hitam). Area arsiran (terang - gelap) jadi punya nilai.
    # Terakhir kita Invert (255 - hasil) supaya warna kertas balik jadi putih (255).
    normalized = 255 - cv2.absdiff(background, warped_gray)

    # 3. PERTAJAM ARSIRAN (CLAHE)
    # Karena bayangan ekstrem udah hilang, CLAHE sekarang bisa fokus 
    # murni mempertajam arsiran pensil vs kertas putih.
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 4. Light blur untuk menghaluskan noise kertas
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
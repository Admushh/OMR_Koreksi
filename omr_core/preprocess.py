import cv2
import numpy as np

def preprocess_for_markers(image):
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. CLAHE — normalize lighting across the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Gaussian Blur 
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

# 4. Adaptive Thresholding 
    thresh = cv2.adaptiveThreshold(
        blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        75,  
        15   
    )

    # 5. Morphological Open — remove small noise specks
    kernel_noise = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise)

    return thresh

def preprocess_for_answers(warped_gray):

    if len(warped_gray.shape) == 3:
        warped_gray = cv2.cvtColor(warped_gray, cv2.COLOR_BGR2GRAY)

    # 1. ESTIMASI BACKGROUND (Menangkap Pola Bayangan)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    
    # DILATE: Algoritma ini akan menghilangkan semua objek gelap (tinta/pensil),   
    background = cv2.morphologyEx(warped_gray, cv2.MORPH_DILATE, kernel)
    
    # Blur sedikit agar transisi bayangannya mulus
    background = cv2.medianBlur(background, 21)

    # 2. HAPUS BAYANGAN (Subtraksi)
    normalized = 255 - cv2.absdiff(background, warped_gray)

    # (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 4. Light blur untuk menghaluskan noise kertas
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced

def preprocess_image(image):

    return preprocess_for_markers(image)
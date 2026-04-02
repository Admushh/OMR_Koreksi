import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess gambar OMR dengan CLAHE untuk mengatasi pencahayaan buruk,
    tetap mempertahankan strategi proteksi marker sudut.
    """
    # 1. Convert ke Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    # ==========================================================

    # 2. Gaussian Blur (Penting! CLAHE kadang bikin noise naik, ini buat ngeredam)
    blur = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    
    # 3. Adaptive Threshold (Sekarang inputnya 'blur' hasil CLAHE)  
    thresh = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11,  # Block size
        2    # C constant
    )
    
    # --- SISA KODE KE BAWAH SAMA PERSIS (PROTEKSI SUDUT & HAPUS GARIS) ---
    
    h, w = thresh.shape
    
    # Proteksi Area Sudut (15%)
    corner_margin = 0.15
    margin_h = int(h * corner_margin)
    margin_w = int(w * corner_margin)
    
    mask_center = np.zeros_like(thresh)
    mask_center[margin_h:h-margin_h, margin_w:w-margin_w] = 255
    
    thresh_center = cv2.bitwise_and(thresh, mask_center)
    
    # Deteksi Garis Horizontal & Vertikal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Gabung & Hapus Garis
    detected_lines = cv2.add(detect_horizontal, detect_vertical)
    thresh_cleaned = cv2.subtract(thresh, detected_lines)
    
    # Finishing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_final = cv2.dilate(thresh_cleaned, kernel, iterations=1)
    
    return thresh_final
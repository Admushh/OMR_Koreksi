import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess OMR image (IMPROVED VERSION)
    Fokus: minim noise + bubble tetap jelas
    """

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE 
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 3. Blur (Gaussian lebih stabil buat dokumen)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # ==========================================
    # 4. THRESHOLD (GANTI KE OTSU)
    # ==========================================
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ==========================================
    # 5. NOISE REMOVAL (PENTING BANGET)
    # ==========================================
    kernel_noise = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise)

    # ==========================================
    # 6. PROTEKSI SUDUT
    # ==========================================
    h, w = thresh.shape
    margin = 0.15

    margin_h = int(h * margin)
    margin_w = int(w * margin)

    mask_center = np.zeros_like(thresh)
    mask_center[margin_h:h-margin_h, margin_w:w-margin_w] = 255

    thresh_center = cv2.bitwise_and(thresh, mask_center)

    # ==========================================
    # 7. DETECT GARIS
    # ==========================================
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    detect_horizontal = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detect_vertical = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    detected_lines = cv2.add(detect_horizontal, detect_vertical)

    # ==========================================
    # 8. HAPUS GARIS
    # ==========================================
    thresh_cleaned = cv2.subtract(thresh, detected_lines)

    # ==========================================
    # 9. FINAL TOUCH (DILATE)
    # ==========================================
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_final = cv2.dilate(thresh_cleaned, kernel, iterations=1)

    return thresh_final
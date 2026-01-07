import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess gambar OMR dengan strategi yang lebih hati-hati
    untuk mempertahankan marker sudut
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur untuk mengurangi noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Adaptive Threshold - Parameter yang lebih konservatif
    thresh = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11,  # Block size lebih kecil (dari 19 ke 11)
        2    # C constant lebih kecil (dari 5 ke 2)
    )
    
    # 3. STRATEGI BARU: Proteksi Area Sudut Sebelum Hapus Garis
    h, w = thresh.shape
    
    # Buat mask untuk melindungi area sudut (15% dari tiap sisi)
    corner_margin = 0.15
    margin_h = int(h * corner_margin)
    margin_w = int(w * corner_margin)
    
    # Mask untuk area yang AKAN dihapus garisnya (tengah saja)
    mask_center = np.zeros_like(thresh)
    mask_center[margin_h:h-margin_h, margin_w:w-margin_w] = 255
    
    # Copy untuk deteksi garis HANYA di area tengah
    thresh_center = cv2.bitwise_and(thresh, mask_center)
    
    # 4. HAPUS GARIS HORIZONTAL (Hanya di tengah)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, 
                                         horizontal_kernel, iterations=2)
    
    # 5. HAPUS GARIS VERTIKAL (Hanya di tengah)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, 
                                       vertical_kernel, iterations=2)
    
    # 6. Gabungkan semua garis yang terdeteksi
    detected_lines = cv2.add(detect_horizontal, detect_vertical)
    
    # 7. Hapus garis dari gambar asli
    thresh_cleaned = cv2.subtract(thresh, detected_lines)
    
    # 8. Sedikit dilate untuk memperkuat marker & bubble
    # TAPI kernel lebih kecil agar tidak merging marker dengan tabel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_final = cv2.dilate(thresh_cleaned, kernel, iterations=1)
    
    return thresh_final


def preprocess_image_simple(image):
    """
    Versi alternatif TANPA line removal - untuk testing
    Gunakan ini jika versi utama masih gagal
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold lebih agresif untuk pastikan marker terlihat
    thresh = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    # Sedikit morphology untuk clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return thresh


# ============================================================
# NEW FUNCTIONS - Add these for the two-stage preprocessing
# ============================================================

def preprocess_for_detection(image):
    """
    Preprocessing khusus untuk MARKER DETECTION
    Prioritas: Jaga corner markers tetap utuh
    
    This is just an alias for preprocess_image_simple
    """
    return preprocess_image_simple(image)


def preprocess_for_grading(warped_image):
    """
    Preprocessing khusus untuk BUBBLE DETECTION setelah warping
    STRATEGI BARU: JANGAN hapus garis, biarkan detect_answers yang handle!
    
    Input: Warped image (sudah 1000x1414)
    Output: Simple threshold untuk bubble detection
    """
    # Jika sudah grayscale (dari warping threshold image)
    if len(warped_image.shape) == 2:
        # Sudah binary/threshold dari warping, return as-is
        return warped_image
    else:
        # Convert ke grayscale dan threshold
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return thresh
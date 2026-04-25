import cv2
import numpy as np

def preprocess_debug(image):
    """
    Preprocess OMR image (IMPROVED VERSION)
    Fokus: minim noise + bubble tetap jelas
    """
    steps = [] 

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps.append(("1. Gray", gray))

    # 2. CLAHE 
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    steps.append(("2. CLAHE", enhanced))

    # 3. Blur (Gaussian lebih stabil buat dokumen)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    steps.append(("3. Blur", blur))

    # ==========================================
    # 4. THRESHOLD
    # ==========================================
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    steps.append(("4. OTSU", thresh))

    # ==========================================
    # 5. NOISE REMOVAL 
    # ==========================================
    kernel_noise = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise)
    steps.append(("5. Clean Noise", thresh))

    # ==========================================
    # 6. PROTEKSI SUDUT
    h, w = thresh.shape
    margin = 0.15

    margin_h = int(h * margin)
    margin_w = int(w * margin)

    mask_center = np.zeros_like(thresh)
    mask_center[margin_h:h-margin_h, margin_w:w-margin_w] = 255

    thresh_center = cv2.bitwise_and(thresh, mask_center)

    # 7. DETECT GARIS
    # ==========================================
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    detect_horizontal = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detect_vertical = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    detected_lines = cv2.add(detect_horizontal, detect_vertical)
    steps.append(("6. Detect Lines", detected_lines))

    # ==========================================
    # 8. HAPUS GARIS
    # ==========================================
    thresh_cleaned = cv2.subtract(thresh, detected_lines)
    steps.append(("7. Lines Removed", thresh_cleaned))

    # ==========================================
    # 9. FINAL TOUCH (DILATE)
    # ==========================================
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_final = cv2.dilate(thresh_cleaned, kernel, iterations=1)
    steps.append(("8. Final", thresh_final))

    # Balikin isi listnya, bukan cuma 1 gambar
    return steps
def show_grid(thresh_final):
    images = []

    for name, img in thresh_final:
        # pastikan semua jadi BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # resize FIX SIZE (WAJIB!)
        img = cv2.resize(img, (300, 300))

        # copy biar gak overwrite memory
        img = img.copy()

        # kasih label
        cv2.putText(img, name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        images.append(img)

    # pastikan jumlah cukup
    while len(images) < 8:
        images.append(np.zeros((300,300,3), dtype=np.uint8))

    row1 = np.hstack(images[:4])
    row2 = np.hstack(images[4:8])

    grid = np.vstack([row1, row2])

    cv2.namedWindow("GRID", cv2.WINDOW_NORMAL)
    cv2.imshow("GRID", grid)

# =========================
# MAIN
# =========================
img = cv2.imread("sample 2.png")

steps = preprocess_debug(img)
show_grid(steps)

cv2.waitKey(0)
cv2.destroyAllWindows() 
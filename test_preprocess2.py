import cv2
import numpy as np

def preprocess_debug_adaptive(image):
    """
    Versi Adaptive Threshold untuk PERBANDINGAN.
    Tujuan: Menunjukkan noise dan hollow bubble jika tidak pakai CLAHE+OTSU.
    """
    steps = []

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps.append(("1. Gray", gray))

    # 2. Blur (Wajib sebelum Adaptive biar pori kertas agak halus)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    steps.append(("2. Blur", blur))

    # ==========================================
    # 3. ADAPTIVE THRESHOLD (Pengganti CLAHE + OTSU)
    # ==========================================
    # Pakai block size standar (misal 21) untuk memancing noise keluar
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )
    steps.append(("3. Adaptive", thresh))

    # ==========================================
    # 4. NOISE REMOVAL 
    # ==========================================
    kernel_noise = np.ones((3,3), np.uint8)
    clean_noise = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise)
    steps.append(("4. Clean Noise", clean_noise))

    # ==========================================
    # 5. PROTEKSI SUDUT & MASKING
    # ==========================================
    h, w = clean_noise.shape
    margin = 0.15

    margin_h = int(h * margin)
    margin_w = int(w * margin)

    mask_center = np.zeros_like(clean_noise)
    mask_center[margin_h:h-margin_h, margin_w:w-margin_w] = 255

    thresh_center = cv2.bitwise_and(clean_noise, mask_center)

    # ==========================================
    # 6. DETECT & HAPUS GARIS
    # ==========================================
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    detect_horizontal = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detect_vertical = cv2.morphologyEx(thresh_center, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    detected_lines = cv2.add(detect_horizontal, detect_vertical)
    steps.append(("5. Detect Lines", detected_lines))

    thresh_cleaned = cv2.subtract(clean_noise, detected_lines)
    steps.append(("6. Lines Removed", thresh_cleaned))

    # ==========================================
    # 7. FINAL TOUCH (DILATE)
    # ==========================================
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_final = cv2.dilate(thresh_cleaned, kernel, iterations=1)
    steps.append(("7. Final Adaptive", thresh_final))

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
img = cv2.imread("Kunjab.jpg")

steps = preprocess_debug_adaptive(img)
show_grid(steps)

cv2.waitKey(0)
cv2.destroyAllWindows() 
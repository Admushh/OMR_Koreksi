import cv2
import sys
import numpy as np
# Pastikan struktur folder omr_core sudah benar
try:
    from omr_core.preprocess import preprocess_image
    from omr_core.detect_sheet import find_paper
except ImportError as e:
    print("Error Import:", e)
    print("Pastikan script ini ada di folder luar (sejajar dengan folder 'omr_core')")
    sys.exit()

# ==========================================
# FUNGSI BANTUAN (LOCAL SIMPLE PREPROCESS)
# ==========================================
def preprocess_simple_local(image):
    """Metode sederhana untuk pembanding (Tanpa Hapus Garis & Tanpa CLAHE)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    return thresh

def resize_for_display(image, max_height=800):
    """Resize image agar muat di layar laptop"""
    if image is None: return None
    
    if len(image.shape) == 2:  # Grayscale
        h, w = image.shape
    else:  # Color
        h, w, _ = image.shape
    
    if h > max_height:
        ratio = max_height / h
        new_w = int(w * ratio)
        return cv2.resize(image, (new_w, max_height))
    return image

# ==========================================
# MAIN PROGRAM
# ==================j========================

# 1. Baca Gambar
filename = "sample.jpg"  # Ganti dengan nama file fotomu
img = cv2.imread(filename)

if img is None:
    print(f"ERROR: File '{filename}' tidak ditemukan!")
    sys.exit()

print("="*60)
print("OMR DETECTION TEST - DEBUG MODE")
print("="*60)

# 2. Preprocess dengan 2 metode
print("\n[1/4] Running Preprocessing...")

# Method 1: Menggunakan fungsi utama kamu (CLAHE + Line Removal)
print("   > Method 1: Advanced (CLAHE + Line Removal)...")
thresh1 = preprocess_image(img)
cv2.imwrite("debug_1_thresh_advanced.png", thresh1)

# Method 2: Menggunakan fungsi lokal sederhana
print("   > Method 2: Simple (Standard Threshold)...")
thresh2 = preprocess_simple_local(img)
cv2.imwrite("debug_2_thresh_simple.png", thresh2)


# 3. Test Detection (Method 1 - Advanced)
print("\n[2/4] Testing Detection on Method 1 (Advanced)...")
debug_img1 = img.copy()

try:
    # Asumsi find_paper menerima argumen (threshold, debug_image)
    warped1 = find_paper(thresh1, debug_image=debug_img1)
    
    if warped1 is not None:
        print("   ✓ Detection SUCCESS")
        cv2.imwrite("debug_3_detection_viz_adv.png", debug_img1)
        cv2.imwrite("debug_4_warped_result_adv.png", warped1)
    else:
        print("   X Detection FAILED (Paper not found)")
except Exception as e:
    print(f"   X Error in find_paper: {e}")
    warped1 = None


# 4. Test Detection (Method 2 - Simple)
print("\n[3/4] Testing Detection on Method 2 (Simple)...")
debug_img2 = img.copy()

try:
    warped2 = find_paper(thresh2, debug_image=debug_img2)
    
    if warped2 is not None:
        print("   ✓ Detection SUCCESS")
        cv2.imwrite("debug_5_detection_viz_simple.png", debug_img2)
        cv2.imwrite("debug_6_warped_result_simple.png", warped2)
    else:
        print("   X Detection FAILED (Paper not found)")
except Exception as e:
    print(f"   X Error in find_paper: {e}")
    warped2 = None


# 5. Display Results
print("\n" + "="*60)
print("CHECK YOUR FOLDER FOR SAVED IMAGES:")
print("="*60)
print("1. debug_1_thresh_advanced.png      : Hasil CLAHE & Hapus Garis")
print("2. debug_3_detection_viz_adv.png    : Kotak hijau/merah deteksi")
print("3. debug_4_warped_result_adv.png    : Hasil crop (Harus tegak lurus)")
print("-" * 30)
print("Bandingkan dengan hasil Simple (debug_5 & debug_6).")
print("Metode Advanced harusnya lebih bersih dari bayangan.")

print("\nDisplaying results (Press any key to exit)...")

# Tampilkan jendela (Hanya jika hasil ada)
cv2.imshow("Original", resize_for_display(img))

# Tampilkan Threshold Advanced
if thresh1 is not None:
    cv2.imshow("Threshold Advanced (CLAHE)", resize_for_display(thresh1))

# Tampilkan Hasil Deteksi Advanced
if warped1 is not None:
    cv2.imshow("Detected Contour (Adv)", resize_for_display(debug_img1))
    cv2.imshow("Warped Result (Adv)", resize_for_display(warped1))
else:
    print("! Advanced method failed to warp image.")

cv2.waitKey(0)
cv2.destroyAllWindows()
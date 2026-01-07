import cv2
import sys
from omr_core.preprocess import preprocess_image, preprocess_image_simple
from omr_core.detect_sheet import find_paper

# 1. Baca Gambar
filename = "LJK_30.png" 
img = cv2.imread(filename)

if img is None:
    print(f"ERROR: File '{filename}' tidak ditemukan!")
    sys.exit()

print("="*60)
print("OMR MARKER DETECTION - DEBUG MODE")
print("="*60)

# 2. Preprocess dengan 2 metode
print("\n--- Method 1: With Line Removal ---")
thresh1 = preprocess_image(img)
cv2.imwrite("debug_thresh_with_removal.png", thresh1)
print("✓ Saved: debug_thresh_with_removal.png")

print("\n--- Method 2: Simple (No Line Removal) ---")
thresh2 = preprocess_image_simple(img)
cv2.imwrite("debug_thresh_simple.png", thresh2)
print("✓ Saved: debug_thresh_simple.png")

# 3. Test detection dengan Method 1
print("\n" + "="*60)
print("TESTING METHOD 1 (With Line Removal)")
print("="*60)
debug_img1 = img.copy()
warped1 = find_paper(thresh1, debug_image=debug_img1)
cv2.imwrite("debug_detection_method1.png", debug_img1)
cv2.imwrite("debug_warped_method1.png", warped1)

# 4. Test detection dengan Method 2
print("\n" + "="*60)
print("TESTING METHOD 2 (Simple)")
print("="*60)
debug_img2 = img.copy()
warped2 = find_paper(thresh2, debug_image=debug_img2)
cv2.imwrite("debug_detection_method2.png", debug_img2)
cv2.imwrite("debug_warped_method2.png", warped2)

# 5. Display Results
print("\n" + "="*60)
print("SAVED DEBUG FILES:")
print("="*60)
print("1. debug_thresh_with_removal.png  - Threshold result (method 1)")
print("2. debug_thresh_simple.png        - Threshold result (method 2)")
print("3. debug_detection_method1.png    - Detection visualization (method 1)")
print("4. debug_detection_method2.png    - Detection visualization (method 2)")
print("5. debug_warped_method1.png       - Final warped result (method 1)")
print("6. debug_warped_method2.png       - Final warped result (method 2)")
print("\nCHECK THESE FILES:")
print("- Do you see 4 BLACK SQUARES in the threshold images?")
print("- Are they SOLID and CLEAR?")
print("- Check detection images for colored boxes")
print("- Warped images should be straight 1000x1414 sheets")
print("="*60)

# FIX: Resize images untuk display agar muat di layar
def resize_for_display(image, max_height=800):
    """Resize image untuk ditampilkan di layar"""
    if len(image.shape) == 2:  # Grayscale
        h, w = image.shape
        channels = 1
    else:  # Color
        h, w, channels = image.shape
    
    if h > max_height:
        ratio = max_height / h
        new_w = int(w * ratio)
        resized = cv2.resize(image, (new_w, max_height))
        return resized
    return image

print("\nDisplaying results in windows (resized for screen)...")
print("Press any key to close windows...")

cv2.imshow("1. Original Image", resize_for_display(img))
cv2.imshow("2. Threshold (Simple)", resize_for_display(thresh2))
cv2.imshow("3. Detection Markers", resize_for_display(debug_img2))
cv2.imshow("4. Final Warped Result", resize_for_display(warped2))

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✅ Done! Check the saved PNG files for full resolution images.")
import cv2
import numpy as np
import os
import argparse

def debug_preprocess_for_markers(image_path, output_dir):


    print(f"--- Debugging Preprocess for Markers on {image_path} ---")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]

    # 1. Original
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_0_original.jpg"), image)

    # 2. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_1_gray.jpg"), gray)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_2_clahe.jpg"), enhanced)

    # 4. Gaussian Blur
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_3_blur.jpg"), blur)

  # 4. Adaptive Thresholding — kebal terhadap bayangan (shadow)
    # Block Size: 75 (Area lokalisasi, wajib ganjil. Agak gede biar marker tetap solid)
    # C: 15 (Konstanta pengurang untuk membersihkan noise sisa bayangan)
    thresh = cv2.adaptiveThreshold(
        blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        75,  
        15   
    )
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_4_adaptive_thresh.jpg"), thresh)

    # 6. Morphological Open
    kernel_noise = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_5_morph_open.jpg"), opened)

    print(f"Done. Check the '{output_dir}' directory for step-by-step images.")


def debug_preprocess_for_answers(image_path, output_dir):
    print(f"--- Debugging Preprocess for Answers on {image_path} ---")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]

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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 4. Light blur untuk menghaluskan noise kertas
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced

    print(f"Done. Check the '{output_dir}' directory for step-by-step images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Preprocessing steps step-by-step.")
    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument("--mode", choices=['markers', 'answers', 'both'], default='both', help="Which pipeline to debug")
    parser.add_argument("--out", default="debug_output", help="Output directory for debug images")
    
    args = parser.parse_args()
    
    if args.mode in ['markers', 'both']:
        debug_preprocess_for_markers(args.image_path, args.out)
    if args.mode in ['answers', 'both']:
        debug_preprocess_for_answers(args.image_path, args.out)

"""
debug_visualizer.py — 10-Grid OMR Visualizer (INTEGRATED)
============================================================
Ini adalah versi profesional. Script ini TIDAK menulis ulang
logika, melainkan meng-IMPORT langsung fungsi produksi dari
folder omr_core lu. Dijamin 100% akurat dengan hasil aslinya.
"""

import cv2
import numpy as np
import os

# 🚨 IMPORT LANGSUNG DARI CORE PRODUKSI LU 🚨
from omr_core.preprocess import preprocess_for_markers, preprocess_for_answers
from omr_core.detect_sheet import find_paper
from omr_core.detect_answers import detect_answers

def process_full_debug(image):
    steps = [] 
    original_bgr = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    steps.append(("1. Original Image", original_bgr))

    # =========================================================
    # TAHAP 1: PREPROCESS MARKER
    # =========================================================
    # Visualisasi tengah jalan untuk dipajang di grid
    clahe_m = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    blur_m = cv2.GaussianBlur(clahe_m.apply(gray), (5, 5), 0)
    steps.append(("2. M: CLAHE + Blur", blur_m))

    # EKSEKUSI FUNGSI PRODUKSI LU!
    thresh_clean = preprocess_for_markers(image)
    steps.append(("3. M: Final Biner", thresh_clean))

    # =========================================================
    # TAHAP 2: DETEKSI SUDUT & WARP
    # =========================================================
    debug_corners = original_bgr.copy()
    
    # EKSEKUSI FUNGSI SAKTI LU (Dijamin gak bakal kegocek angka 11 lagi)
    detect_result = find_paper(thresh_clean, debug_image=debug_corners)
    steps.append(("4. M: Detect Corners", debug_corners))

    if detect_result is not None:
        warped_biner, M_warp = detect_result
        steps.append(("5. M: Warped Biner", warped_biner))
        
        # Warp gambar grayscale asli pakai matriks M_warp produksi
        warped_gray = cv2.warpPerspective(gray, M_warp, (1000, 1414))
        steps.append(("6. A: Warped Gray", warped_gray))

        # =========================================================
        # TAHAP 3: PREPROCESS JAWABAN
        # =========================================================
        # Visualisasi Peta Bayangan (Dilakukan pada gambar yang sudah di-warp!)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        background = cv2.medianBlur(cv2.morphologyEx(warped_gray, cv2.MORPH_DILATE, kernel_bg), 21)
        steps.append(("7. A: BG Estimate", background))

        normalized = 255 - cv2.absdiff(background, warped_gray)
        steps.append(("8. A: BG Subtracted", normalized))

        # EKSEKUSI FUNGSI PRODUKSI LU!
        warped_ready = preprocess_for_answers(warped_gray)
        steps.append(("9. A: Final Grayscale", warped_ready))

        # =========================================================
        # TAHAP 4: DETEKSI JAWABAN (Z-SCORE)
        # =========================================================
        # Panggil fungsi produksi dengan mode debug nyala
        detect_answers(warped_ready, num_questions=30, debug=True)
        
        # Karena fungsi lu nge-save gambar overlay ke disk, kita baca balik aja
        if os.path.exists("debug_grid_overlay.png"):
            final_res = cv2.imread("debug_grid_overlay.png")
            steps.append(("10. Final Result", final_res))
        else:
            steps.append(("10. Final Result", warped_ready))
            
    else:
        # Kalau gagal, kosongin kotak sisanya
        steps.append(("5. M: FAIL (No Corners)", np.zeros_like(thresh_clean)))
        for i in range(6, 11):
            steps.append((f"SKIP STEP {i}", np.zeros_like(gray)))

    return steps

# ==========================================
# RENDER GRID (10 KOTAK)
# ==========================================
def show_grid(steps):
    images = []
    for name, img in steps[:10]: # Pastikan cuma 10
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (300, 420))
        img = img.copy()

        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 3)
        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        images.append(img)

    while len(images) < 10:
        images.append(np.zeros((420, 300, 3), dtype=np.uint8))

    row1 = np.hstack(images[:5])
    row2 = np.hstack(images[5:10])
    grid = np.vstack([row1, row2])

    cv2.namedWindow("END-TO-END OMR PIPELINE", cv2.WINDOW_NORMAL)
    cv2.imshow("END-TO-END OMR PIPELINE", grid)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    img = cv2.imread("sample.png") # UBAH NAMA FILE GAMBAR LU DI SINI
    
    if img is not None:
        steps = process_full_debug(img)
        show_grid(steps)
        print("Tekan tombol apapun pada jendela gambar untuk menutup...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Gambar tidak ditemukan! Pastikan path/nama file benar.")
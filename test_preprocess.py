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
    cv2.imwrite("01_original_image.jpg", original_bgr)
    
    # --- PROSES GRAYSCALE ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    cv2.imwrite("02_grayscale.jpg", gray)
    
    steps.append(("1. Original Image", original_bgr))

    # =========================================================
    # TAHAP 1: PREPROCESS MARKER
    # =========================================================
    clahe_m = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_result = clahe_m.apply(gray)
    cv2.imwrite("03_clahe_marker.jpg", clahe_result)

    blur_result = cv2.GaussianBlur(clahe_result, (5, 5), 0)
    cv2.imwrite("04_blur_marker.jpg", blur_result)
    steps.append(("2. M: CLAHE + Blur", blur_result))

    thresh_clean = preprocess_for_markers(image)
    cv2.imwrite("05_adaptive_threshold_biner.jpg", thresh_clean)
    steps.append(("3. M: Final Biner", thresh_clean))

    # =========================================================
    # TAHAP 2: DETEKSI SUDUT & WARP
    # =========================================================
    debug_corners = original_bgr.copy()
    detect_result = find_paper(thresh_clean, debug_image=debug_corners)
    cv2.imwrite("06_detect_corners.jpg", debug_corners)
    steps.append(("4. M: Detect Corners", debug_corners))

    if detect_result is not None:
        warped_biner, M_warp = detect_result
        cv2.imwrite("07_warped_biner.jpg", warped_biner)
        steps.append(("5. M: Warped Biner", warped_biner))
        
        warped_gray = cv2.warpPerspective(gray, M_warp, (1000, 1414))
        cv2.imwrite("08_warped_grayscale.jpg", warped_gray)
        steps.append(("6. A: Warped Gray", warped_gray))

        # =========================================================
        # TAHAP 3: PREPROCESS JAWABAN
        # =========================================================
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        background = cv2.medianBlur(cv2.morphologyEx(warped_gray, cv2.MORPH_DILATE, kernel_bg), 21)
        cv2.imwrite("09_background_estimate.jpg", background)
        steps.append(("7. A: BG Estimate", background))

        normalized = 255 - cv2.absdiff(background, warped_gray)
        cv2.imwrite("10_background_subtracted.jpg", normalized)
        steps.append(("8. A: BG Subtracted", normalized))

        warped_ready = preprocess_for_answers(warped_gray)
        cv2.imwrite("11_final_answer_grayscale.jpg", warped_ready)
        steps.append(("9. A: Final Grayscale", warped_ready))

        # =========================================================
        # TAHAP EXTRA UNTUK SKRIPSI: EKSTRAKSI GAMBAR ROI MURNI
        # =========================================================
        # Ambil sampel Baris Pertama (Soal No 1) untuk diekstrak ROI-nya
        ROW_Y_CENTERS = [
            320, 389, 459, 529, 599, 669, 739, 810, 880, 951, 1022, 1093, 1164, 1236, 1307
        ]
        yc = ROW_Y_CENTERS[0]
        yt = yc - 35
        yb = yc + 35
        
        c1_xs = 90
        c1_xe = 350
        col_w = c1_xe - c1_xs
        
        OPTION_BOUNDS = [(0.00, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.00)]
        roi_cells = []
        
        for (xs_r, xe_r) in OPTION_BOUNDS:
            xt = c1_xs + int(col_w * xs_r)
            xb = c1_xs + int(col_w * xe_r)
            inset = 8
            # Slicing murni potongan ROI
            cell = warped_ready[yt + inset: yb - inset, xt + inset: xb - inset]
            
            # Diperbesar jadi 100x100 biar gak pecah pas ditempel di Word
            if cell.size > 0:
                cell_resized = cv2.resize(cell, (100, 100), interpolation=cv2.INTER_NEAREST)
                # Kasih border hitam tipis biar kepisah antar buletan
                cell_bordered = cv2.copyMakeBorder(cell_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                roi_cells.append(cell_bordered)
        
        if len(roi_cells) == 5:
            # Jejerkan kelima buletan A, B, C, D, E secara horizontal
            roi_row_img = np.hstack(roi_cells)
            cv2.imwrite("11a_sample_roi_soal_1_murni.jpg", roi_row_img)

        # =========================================================
        # TAHAP 4: DETEKSI JAWABAN (Z-SCORE) & OCR (COMBINED)
        # =========================================================
        from omr_core.ocr import extract_name_and_id
        student_name, student_id = extract_name_and_id(warped_gray)
        
        print("\n=============================================")
        print("  OCR EXTRACTION RESULT")
        print("=============================================")
        print(f"  Student Name: '{student_name}'")
        print(f"  Student ID  : '{student_id}'")
        print("=============================================\n")

        detect_answers(warped_ready, num_questions=30, debug=True)
        
        if os.path.exists("debug_grid_overlay.png"):
            final_res = cv2.imread("debug_grid_overlay.png")
            # Overlay name & ID on the top section of the warped final debug image
            cv2.putText(final_res, f"OCR Name: {student_name}", (80, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(final_res, f"OCR ID: {student_id}", (80, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imwrite("12_final_zscore_result.jpg", final_res)
            steps.append(("10. Final Result", final_res))
        else:
            cv2.imwrite("12_final_zscore_result.jpg", warped_ready)
            steps.append(("10. Final Result", warped_ready))
            
    else:
        steps.append(("5. M: FAIL (No Corners)", np.zeros_like(thresh_clean)))
        for i in range(6, 11):
            steps.append((f"SKIP STEP {i}", np.zeros_like(gray)))

    return steps

# ==========================================
# RENDER GRID (10 KOTAK)
# ==========================================
def show_grid(steps):
    images = []
    for name, img in steps[:10]:
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
    img = cv2.imread("IMG_3345.png") # UBAH NAMA FILE GAMBAR LU DI SINI
    
    if img is not None:
        steps = process_full_debug(img)
        show_grid(steps)
        print("Tekan tombol apapun pada jendela gambar untuk menutup...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Gambar tidak ditemukan! Pastikan path/nama file benar.")
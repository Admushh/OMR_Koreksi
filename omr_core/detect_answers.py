import cv2
import numpy as np

def get_bubble_grid_custom(roi, questions=15, bubble_positions=None, debug_name="", debug_img=None, start_x_global=0, start_y_global=0):
    column_answers = []
    h, w = roi.shape
    
    # 1. Binerisasi Cerdas (Otsu Thresholding)
    # Paksa gambar jadi murni hitam-putih. Background hitam, coretan/bulatan jadi putih.
    # Ini sangat krusial biar gampang ngitung area isian pensil.
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # 2. Hancurkan Garis Tabel (Morphology Open)
    # Ini senjata rahasia lu: Hapus garis tipis horizontal/vertikal tabel, tapi pertahankan bulatan pensil tebal.
    kernel_bulat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_bersih = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_bulat)

    y_steps = np.linspace(0, h, questions + 1).astype(int)

    for q in range(questions):
        y_start = y_steps[q]
        y_end = y_steps[q+1]
        
        # Kasih "ruang bernapas" 5 piksel atas bawah, biar coretan yang mbleber gak kepotong
        overlap = 5
        row_slice = binary_bersih[max(0, y_start-overlap):min(h, y_end+overlap), :]
        
        max_pixels = 0
        bubbled = -1
        all_pixels = []
        
        for c, (start_ratio, end_ratio) in enumerate(bubble_positions):
            x_start = int(w * start_ratio)
            x_end = int(w * end_ratio)
            
            # Potong per kotak opsi (A/B/C/D/E)
            bubble_box = row_slice[:, x_start:x_end]
            
            # Hitung total gumpalan piksel putih di dalam kotak
            white_pixels = cv2.countNonZero(bubble_box)
            all_pixels.append(white_pixels)
            
            if white_pixels > max_pixels:
                max_pixels = white_pixels
                bubbled = c
                
            # --- VISUALISASI DEBUG (KOTAK BIRU) ---
            if debug_img is not None:
                g_x1 = start_x_global + x_start
                g_y1 = start_y_global + max(0, y_start-overlap)
                g_x2 = start_x_global + x_end
                g_y2 = start_y_global + min(h, y_end+overlap)
                cv2.rectangle(debug_img, (g_x1, g_y1), (g_x2, g_y2), (255, 0, 0), 1)

        # Rata-rata area putih dari ke-5 opsi
        avg_pixels = sum(all_pixels) / len(all_pixels) if all_pixels else 0
        
        MIN_FILL_THRESHOLD = 100
        DOUBLE_BUBBLE_RATIO = 0.70  # Jika bubble lain >= 70% dari max, dianggap double
        
        if max_pixels > MIN_FILL_THRESHOLD and max_pixels > (avg_pixels * 1.5):
            # --- CEK DOUBLE BUBBLE ---
            # Cari semua bubble yang "cukup terisi" (>= 70% dari max DAN > threshold)
            filled_bubbles = []
            for idx, px in enumerate(all_pixels):
                if px > MIN_FILL_THRESHOLD and px >= (max_pixels * DOUBLE_BUBBLE_RATIO):
                    filled_bubbles.append(idx)
            
            if len(filled_bubbles) > 1:
                # DOUBLE BUBBLE TERDETEKSI!
                answer = "DOUBLE"
                column_answers.append(answer)
                
                # --- VISUALISASI DEBUG DOUBLE BUBBLE (KOTAK MERAH TEBAL) ---
                if debug_img is not None:
                    for fb_idx in filled_bubbles:
                        gx1 = start_x_global + int(w * bubble_positions[fb_idx][0])
                        gy1 = start_y_global + max(0, y_start-overlap)
                        gx2 = start_x_global + int(w * bubble_positions[fb_idx][1])
                        gy2 = start_y_global + min(h, y_end+overlap)
                        cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)  # MERAH
            else:
                # Single bubble — jawaban valid
                answer = chr(65 + bubbled)
                column_answers.append(answer)
                
                # --- VISUALISASI DEBUG JAWABAN BENAR (KOTAK HIJAU TEBAL) ---
                if debug_img is not None:
                    gx1 = start_x_global + int(w * bubble_positions[bubbled][0])
                    gy1 = start_y_global + max(0, y_start-overlap)
                    gx2 = start_x_global + int(w * bubble_positions[bubbled][1])
                    gy2 = start_y_global + min(h, y_end+overlap)
                    cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)  # HIJAU
        else:
            column_answers.append(None)
            
    return column_answers

def detect_answers(warped_img, num_questions=30, debug=True):
    height, width = warped_img.shape
    
    print(f"\n--- DETECTING ANSWERS (HIGH TOLERANCE) ---")
    
    mean_val = np.mean(warped_img)
    if mean_val < 127:
        warped_img = cv2.bitwise_not(warped_img)

    # === KALIBRASI GRID ===
    bubble_positions = [
        (0.00, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.00)
    ]
    
    # Y-AXIS (Vertikal)
    start_y = int(height * 0.222) 
    end_y = int(height * 0.975)
    
    # X-AXIS (Horizontal)
    col1_start_x = int(width * 0.090) 
    col1_end_x = int(width * 0.350) 
    col2_start_x = int(width * 0.590)
    col2_end_x = int(width * 0.855)

    roi_col1 = warped_img[start_y:end_y, col1_start_x:col1_end_x]
    roi_col2 = warped_img[start_y:end_y, col2_start_x:col2_end_x]
    
    if debug:
        debug_vis = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_vis, (col1_start_x, start_y), (col1_end_x, end_y), (0, 255, 0), 2)
        cv2.rectangle(debug_vis, (col2_start_x, start_y), (col2_end_x, end_y), (0, 255, 0), 2)
        
        questions_per_col = 15
        y_steps = np.linspace(start_y, end_y, questions_per_col + 1).astype(int)
        
        col1_w = col1_end_x - col1_start_x
        for y in y_steps: cv2.line(debug_vis, (col1_start_x, y), (col1_end_x, y), (255, 0, 0), 1)
        for r_start, _ in bubble_positions:
            x = col1_start_x + int(col1_w * r_start)
            cv2.line(debug_vis, (x, start_y), (x, end_y), (255, 0, 0), 1)
            
        col2_w = col2_end_x - col2_start_x
        for y in y_steps: cv2.line(debug_vis, (col2_start_x, y), (col2_end_x, y), (255, 0, 0), 1)
        for r_start, _ in bubble_positions:
            x = col2_start_x + int(col2_w * r_start)
            cv2.line(debug_vis, (x, start_y), (x, end_y), (255, 0, 0), 1)

    answers_part1 = get_bubble_grid_custom(
        roi_col1, questions=15, bubble_positions=bubble_positions,
        debug_name="Col 1", debug_img=debug_vis if debug else None,
        start_x_global=col1_start_x, start_y_global=start_y
    )
                                    
    answers_part2 = get_bubble_grid_custom(
        roi_col2, questions=15, bubble_positions=bubble_positions,
        debug_name="Col 2", debug_img=debug_vis if debug else None,
        start_x_global=col2_start_x, start_y_global=start_y
    )
    
    if debug:
        cv2.imwrite("debug_grid_overlay.png", debug_vis)
    
    all_answers = answers_part1 + answers_part2
    answers_dict = {i: ans for i, ans in enumerate(all_answers, start=1)}
    
    # --- PRINT FULL REPORT UNTUK CROSSCHECK ---
    print(f"\n{'='*40}")
    print(f"REKAP HASIL DETEKSI (RAW)")
    print(f"{'='*40}")
    print(f"{'No':<5} | {'Jawaban Deteksi'}")
    print(f"{'-'*25}")
    
    for i in range(1, 31):
        ans = answers_dict.get(i)
        if ans == "DOUBLE":
            display_ans = "⚠️ [DOUBLE BUBBLE]"
        elif ans is not None:
            display_ans = ans
        else:
            display_ans = "[KOSONG]"
        print(f"Q{i:<4} | {display_ans}")
        
    print(f"{'='*40}\n")
    
    return answers_dict

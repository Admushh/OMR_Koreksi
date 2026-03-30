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
        
        # --- LOGIKA PENENTUAN JAWABAN (ANTI-KOSONG) ---
        # 1. MIN_FILL: Harus ada minimal 120 piksel putih (artinya beneran diarsir, bukan ketumpahan tinta titik)
        # 2. RASIO: Opsi tersebut harus 1.5x (50%) lebih pekat dari rata-rata opsi lain di nomor tersebut.
        MIN_FILL_THRESHOLD = 100
        
        if max_pixels > MIN_FILL_THRESHOLD and max_pixels > (avg_pixels * 1.5):
            answer = chr(65 + bubbled)
            column_answers.append(answer)
            
            # --- VISUALISASI DEBUG JAWABAN BENAR (KOTAK HIJAU TEBAL) ---
            if debug_img is not None:
                gx1 = start_x_global + int(w * bubble_positions[bubbled][0])
                gy1 = start_y_global + max(0, y_start-overlap)
                gx2 = start_x_global + int(w * bubble_positions[bubbled][1])
                gy2 = start_y_global + min(h, y_end+overlap)
                cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
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
        display_ans = ans if ans is not None else "[KOSONG]"
        print(f"Q{i:<4} | {display_ans}")
        
    print(f"{'='*40}\n")
    
    return answers_dict

def grade_answers(student_answers, answer_key):
    correct = 0; wrong = 0; empty = 0; details = {}
    
    # Header Report Grading
    print(f"\n{'='*60}")
    print(f"DETAIL PENILAIAN (GRADING)")
    print(f"{'='*60}")
    print(f"{'No':<4} | {'Siswa':<7} | {'Kunci':<7} | {'Status'}")
    print(f"{'-'*45}")

    for q_num in sorted(student_answers.keys()):
        student_ans = student_answers[q_num]
        correct_ans = answer_key.get(q_num)
        
        status = ""
        if student_ans is None: 
            empty += 1
            status = "EMPTY"
            disp_std = "[ - ]"
        elif student_ans == correct_ans: 
            correct += 1
            status = "CORRECT"
            disp_std = f"  {student_ans}  "
        else: 
            wrong += 1
            status = "WRONG"
            disp_std = f"  {student_ans}  "
            
        details[q_num] = {'student': student_ans, 'correct': correct_ans, 'status': status}
        
        # Print baris per baris agar user bisa cek semua nomor
        print(f"Q{q_num:<3} | {disp_std:<7} |   {correct_ans:<5} | {status}")

    print(f"{'='*60}")

    total = len(student_answers)
    score = (correct / total) * 100 if total > 0 else 0
    return {'Nilai': score, 'Benar': correct, 'Salah': wrong, 'Kosong': empty, 'Total': total, 'Detail': details}
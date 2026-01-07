import cv2
import numpy as np

def get_bubble_grid_custom(roi, questions=15, bubble_positions=None, debug_name="", debug_img=None, start_x_global=0, start_y_global=0):
    column_answers = []
    h, w = roi.shape
    
    # Gunakan np.linspace untuk presisi Vertikal
    y_steps = np.linspace(0, h, questions + 1).astype(int)

    for q in range(questions):
        y_start = y_steps[q]
        y_end = y_steps[q+1]
        row = roi[y_start:y_end, :]
        
        bubbled = None
        max_pixels = 0
        all_pixels = []
        
        for c, (start_ratio, end_ratio) in enumerate(bubble_positions):
            x_start = int(w * start_ratio)
            x_end = int(w * end_ratio)
            col = row[:, x_start:x_end]
            
            # --- ADJUSTMENT KOTAK KUNING (RESIZED) ---
            # Margin 22% (Kotak Kuning Lebar)
            margin_h = int(col.shape[0] * 0.22) 
            margin_w = int(col.shape[1] * 0.22)
            
            # Safety check
            if col.shape[0] > margin_h*2 + 2 and col.shape[1] > margin_w*2 + 2:
                inner_bubble = col[margin_h:-margin_h, margin_w:-margin_w]
            else:
                inner_bubble = col
            
            # PENEBALAN TINTA
            kernel = np.ones((3, 3), np.uint8)
            inner_bubble = cv2.erode(inner_bubble, kernel, iterations=1)

            white_pixels = cv2.countNonZero(inner_bubble)
            black_pixels = inner_bubble.size - white_pixels 
            all_pixels.append(black_pixels)
            
            if black_pixels > max_pixels:
                max_pixels = black_pixels
                bubbled = c
            
            if debug_img is not None:
                g_x1 = start_x_global + x_start + margin_w
                g_y1 = start_y_global + y_start + margin_h
                g_x2 = start_x_global + x_end - margin_w
                g_y2 = start_y_global + y_end - margin_h
                
                if g_x2 > g_x1 and g_y2 > g_y1:
                    cv2.rectangle(debug_img, (g_x1, g_y1), (g_x2, g_y2), (0, 165, 255), 1)

        avg_pixels = sum(all_pixels) / len(all_pixels) if all_pixels else 0
        
        # === ADJUSTMENT TOLERANSI DISINI ===
        # 1. Turunkan batas minimal pixel hitam dari 50 ke 20
        #    Agar jawaban tipis/terpotong sedikit tetap dianggap ada isinya.
        MIN_FILL_THRESHOLD = 20 
        
        if max_pixels < MIN_FILL_THRESHOLD:
            column_answers.append(None)
        # 2. Turunkan rasio perbandingan dari 1.3 ke 1.1
        #    Agar jawaban tidak harus 'sangat kontras' dibanding noise.
        #    Cukup sedikit lebih gelap dari rata-rata, kita anggap jawaban.
        elif max_pixels > avg_pixels * 1.1: 
            answer = chr(65 + bubbled)
            column_answers.append(answer)
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
    return {'score': score, 'correct': correct, 'wrong': wrong, 'empty': empty, 'total': total, 'details': details}
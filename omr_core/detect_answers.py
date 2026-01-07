import cv2
import numpy as np

def get_bubble_grid(roi, questions=15, choices=5, debug_name=""):
    """
    Proses satu blok kolom (misal 1-15 atau 16-30)
    
    Args:
        roi: Region of interest (cropped area)
        questions: Jumlah soal dalam kolom ini
        choices: Jumlah pilihan (A-E = 5)
        debug_name: Nama untuk debug output
    
    Returns:
        List jawaban untuk kolom ini
    """
    column_answers = []
    h, w = roi.shape
    
    # Hitung ukuran per kotak
    box_h = h // questions
    box_w = w // choices
    
    print(f"\n  Processing {debug_name}:")
    print(f"    ROI size: {w}x{h}")
    print(f"    Box size: {box_w}x{box_h} per bubble")

    for q in range(questions):
        # Ambil satu baris soal
        row = roi[q * box_h:(q + 1) * box_h, :]
        
        bubbled = None
        max_pixels = 0
        all_pixels = []
        
        for c in range(choices):
            # Ambil satu bubble
            col = row[:, c * box_w:(c + 1) * box_w]
            
            # PENTING: Potong margin BESAR agar garis tabel tidak terhitung
            # Kita ambil bagian tengah bubble saja (30% margin dari tiap sisi)
            margin_h = int(col.shape[0] * 0.3)
            margin_w = int(col.shape[1] * 0.3)
            
            if col.shape[0] > margin_h*2 and col.shape[1] > margin_w*2:
                col = col[margin_h:-margin_h, margin_w:-margin_w]
            
            # Hitung pixel HITAM (terisi) = 255 - putih
            # Karena background putih, bubble terisi = hitam
            white_pixels = cv2.countNonZero(col)
            total = col.size - white_pixels  # Hitung pixel hitam
            all_pixels.append(total)
            
            if total > max_pixels:
                max_pixels = total
                bubbled = c
        
        # ADAPTIVE THRESHOLD: Bandingkan dengan rata-rata
        # Jika bubble "terpilih" jauh lebih terisi dari yang lain, itu jawaban
        avg_pixels = sum(all_pixels) / len(all_pixels)
        
        # Bubble harus:
        # 1. Punya pixel HITAM > threshold minimum (ada isian)
        # 2. Punya pixel HITAM > 1.5x rata-rata (jelas lebih terisi dari bubble kosong)
        MIN_FILL_THRESHOLD = 100  # Pixel hitam minimum untuk dianggap terisi
        
        if max_pixels < MIN_FILL_THRESHOLD:
            # Terlalu sedikit pixel = kosong
            column_answers.append(None)
        elif max_pixels > avg_pixels * 1.5:
            # Jelas lebih terisi dari yang lain
            column_answers.append(chr(65 + bubbled))  # 65 = 'A'
        else:
            # Ambiguous atau kosong
            column_answers.append(None)
            
    return column_answers


def detect_answers(warped_img, num_questions=30, debug=True):
    """
    Deteksi jawaban dari lembar OMR yang sudah diluruskan dan dibersihkan
    
    Args:
        warped_img: Gambar threshold yang sudah di-warp dan clean (1000x1414)
        num_questions: Total jumlah soal (default 30)
        debug: Simpan debug images atau tidak
    
    Returns:
        answers: Dictionary {question_num: answer} (1-based indexing)
    """
    height, width = warped_img.shape
    
    print(f"\n--- DETECTING ANSWERS ---")
    print(f"Input image size: {width}x{height}")
    
    # CRITICAL FIX: Pastikan image dalam format yang benar
    # Kita butuh: Background PUTIH (255), Bubble terisi HITAM (0)
    # Tapi kadang hasil warping = Background HITAM, Bubble PUTIH
    
    # Check apakah perlu di-invert
    mean_val = np.mean(warped_img)
    print(f"Image mean value: {mean_val:.1f}")
    
    if mean_val < 127:  # Image lebih banyak hitam = salah!
        print("⚠️  Image is inverted (dark background), inverting...")
        warped_img = cv2.bitwise_not(warped_img)
        if debug:
            cv2.imwrite("debug_inverted_fixed.png", warped_img)
    else:
        print("✓ Image format correct (light background)")
    
    # --- KONFIGURASI ROI (REGION OF INTEREST) ---
    # Sesuaikan dengan layout LJK kamu
    
    start_y = int(height * 0.22)  # Skip Header (22% dari atas)
    end_y = int(height * 0.95)    # Margin Bawah (5% dari bawah)
    
    # Kolom Kiri (1-15)
    col1_start_x = int(width * 0.08)
    col1_end_x = int(width * 0.48)
    
    # Kolom Kanan (16-30)
    col2_start_x = int(width * 0.52)
    col2_end_x = int(width * 0.92)
    
    print(f"ROI coordinates:")
    print(f"  Y-range: {start_y} to {end_y}")
    print(f"  Column 1 (Q1-15):  X={col1_start_x} to {col1_end_x}")
    print(f"  Column 2 (Q16-30): X={col2_start_x} to {col2_end_x}")

    # Crop ROI
    roi_col1 = warped_img[start_y:end_y, col1_start_x:col1_end_x]
    roi_col2 = warped_img[start_y:end_y, col2_start_x:col2_end_x]
    
    # --- DEBUGGING ---
    if debug:
        cv2.imwrite("debug_crop_col1.png", roi_col1)
        cv2.imwrite("debug_crop_col2.png", roi_col2)
        print(f"\n✓ Saved debug crops")
        
        # Buat visualisasi dengan garis grid
        debug_vis = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)
        
        # Gambar kotak ROI
        cv2.rectangle(debug_vis, (col1_start_x, start_y), (col1_end_x, end_y), (0, 255, 0), 2)
        cv2.rectangle(debug_vis, (col2_start_x, start_y), (col2_end_x, end_y), (0, 255, 0), 2)
        
        # Gambar grid untuk tiap bubble
        questions_per_col = 15
        box_h = (end_y - start_y) // questions_per_col
        box_w1 = (col1_end_x - col1_start_x) // 5
        box_w2 = (col2_end_x - col2_start_x) // 5
        
        # Grid kolom 1
        for i in range(questions_per_col + 1):
            y = start_y + i * box_h
            cv2.line(debug_vis, (col1_start_x, y), (col1_end_x, y), (255, 0, 0), 1)
        for i in range(6):
            x = col1_start_x + i * box_w1
            cv2.line(debug_vis, (x, start_y), (x, end_y), (255, 0, 0), 1)
        
        # Grid kolom 2
        for i in range(questions_per_col + 1):
            y = start_y + i * box_h
            cv2.line(debug_vis, (col2_start_x, y), (col2_end_x, y), (255, 0, 0), 1)
        for i in range(6):
            x = col2_start_x + i * box_w2
            cv2.line(debug_vis, (x, start_y), (x, end_y), (255, 0, 0), 1)
        
        cv2.imwrite("debug_grid_overlay.png", debug_vis)
        print(f"✓ Saved grid visualization")

    # Deteksi bubble untuk setiap kolom
    answers_part1 = get_bubble_grid(roi_col1, questions=15, choices=5, debug_name="Column 1 (Q1-15)")
    answers_part2 = get_bubble_grid(roi_col2, questions=15, choices=5, debug_name="Column 2 (Q16-30)")
    
    # Gabungkan dan buat dictionary dengan numbering 1-based
    all_answers = answers_part1 + answers_part2
    
    answers_dict = {}
    for i, ans in enumerate(all_answers, start=1):
        answers_dict[i] = ans
    
    # Print hasil deteksi
    print(f"\n📋 Detection Results:")
    for q_num, answer in answers_dict.items():
        status = answer if answer else "(kosong)"
        print(f"  Q{q_num:2d}: {status}")
    
    return answers_dict


def grade_answers(student_answers, answer_key):
    """
    Nilai jawaban siswa
    
    Args:
        student_answers: Dict dari detect_answers() {q_num: answer}
        answer_key: Dict {q_num: correct_answer}
        
    Returns:
        score_info: Dict dengan detail nilai
    """
    correct = 0
    wrong = 0
    empty = 0
    
    details = {}
    
    for q_num in sorted(student_answers.keys()):
        student_ans = student_answers[q_num]
        correct_ans = answer_key.get(q_num)
        
        if student_ans is None:
            empty += 1
            status = "EMPTY"
        elif student_ans == correct_ans:
            correct += 1
            status = "CORRECT"
        else:
            wrong += 1
            status = "WRONG"
        
        details[q_num] = {
            'student': student_ans,
            'correct': correct_ans,
            'status': status
        }
    
    total_questions = len(student_answers)
    score = (correct / total_questions) * 100 if total_questions > 0 else 0
    
    return {
        'score': score,
        'correct': correct,
        'wrong': wrong,
        'empty': empty,
        'total': total_questions,
        'details': details
    }
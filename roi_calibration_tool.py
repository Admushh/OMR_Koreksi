"""
ROI CALIBRATION TOOL
====================
Tool untuk menyesuaikan koordinat ROI agar pas dengan bubble
"""

import cv2
import numpy as np

def draw_roi_grid(image_path, show_interactive=True):
    """
    Gambar grid ROI di atas warped image untuk kalibrasi
    """
    # Load warped image
    warped = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if warped is None:
        print(f"Error: Cannot load {image_path}")
        return
    
    height, width = warped.shape
    
    # Convert to color untuk visualisasi
    vis = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    
    # ============================================================
    # KOORDINAT ROI - SESUAIKAN INI SAMPAI PAS!
    # ============================================================
    
    # Vertikal (Y-axis)
    start_y = int(height * 0.22)  # Mulai dari mana? Coba ubah 0.20-0.25
    end_y = int(height * 0.95)    # Akhir di mana? Coba ubah 0.92-0.97
    
    # Kolom Kiri (X-axis)
    col1_start_x = int(width * 0.08)  # Kiri kolom 1
    col1_end_x = int(width * 0.48)    # Kanan kolom 1
    
    # Kolom Kanan (X-axis)
    col2_start_x = int(width * 0.52)  # Kiri kolom 2
    col2_end_x = int(width * 0.92)    # Kanan kolom 2
    
    # ============================================================
    
    print("\n" + "="*70)
    print("ROI CALIBRATION TOOL")
    print("="*70)
    print(f"Image size: {width}x{height}")
    print(f"\nCurrent ROI coordinates:")
    print(f"  Vertical range:  Y = {start_y} to {end_y}")
    print(f"  Column 1 (Q1-15):  X = {col1_start_x} to {col1_end_x}")
    print(f"  Column 2 (Q16-30): X = {col2_start_x} to {col2_end_x}")
    
    # Hitung parameter grid
    questions_per_col = 15
    choices = 5
    
    row_height = (end_y - start_y) // questions_per_col
    col1_bubble_width = (col1_end_x - col1_start_x) // choices
    col2_bubble_width = (col2_end_x - col2_start_x) // choices
    
    print(f"\nGrid parameters:")
    print(f"  Row height: {row_height}px")
    print(f"  Column 1 bubble width: {col1_bubble_width}px")
    print(f"  Column 2 bubble width: {col2_bubble_width}px")
    
    # Gambar ROI boxes (outer boundary)
    cv2.rectangle(vis, (col1_start_x, start_y), (col1_end_x, end_y), (0, 255, 0), 3)
    cv2.rectangle(vis, (col2_start_x, start_y), (col2_end_x, end_y), (0, 255, 0), 3)
    
    # Gambar grid horizontal (baris soal)
    for i in range(questions_per_col + 1):
        y = start_y + i * row_height
        
        # Garis di kolom 1
        cv2.line(vis, (col1_start_x, y), (col1_end_x, y), (255, 0, 0), 1)
        
        # Garis di kolom 2
        cv2.line(vis, (col2_start_x, y), (col2_end_x, y), (255, 0, 0), 1)
        
        # Label nomor soal
        if i < questions_per_col:
            q_num_col1 = i + 1
            q_num_col2 = i + 16
            cv2.putText(vis, str(q_num_col1), (col1_start_x - 30, y + row_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis, str(q_num_col2), (col2_start_x - 30, y + row_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Gambar grid vertikal (kolom bubble A-E) untuk kolom 1
    for i in range(choices + 1):
        x = col1_start_x + i * col1_bubble_width
        cv2.line(vis, (x, start_y), (x, end_y), (255, 0, 0), 1)
        
        # Label A-E
        if i < choices:
            label = chr(65 + i)  # A, B, C, D, E
            cv2.putText(vis, label, (x + col1_bubble_width//2 - 5, start_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Gambar grid vertikal untuk kolom 2
    for i in range(choices + 1):
        x = col2_start_x + i * col2_bubble_width
        cv2.line(vis, (x, start_y), (x, end_y), (255, 0, 0), 1)
        
        # Label A-E
        if i < choices:
            label = chr(65 + i)
            cv2.putText(vis, label, (x + col2_bubble_width//2 - 5, start_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Highlight beberapa bubble untuk referensi
    # Q1-A (should be filled)
    highlight_bubble(vis, col1_start_x, start_y, col1_bubble_width, row_height, 0, 0, (0, 255, 255))
    # Q2-D (should be filled)
    highlight_bubble(vis, col1_start_x, start_y, col1_bubble_width, row_height, 1, 3, (0, 255, 255))
    # Q16-A (should be filled)
    highlight_bubble(vis, col2_start_x, start_y, col2_bubble_width, row_height, 0, 0, (0, 255, 255))
    
    # Save
    output_path = "roi_calibration.png"
    cv2.imwrite(output_path, vis)
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "="*70)
    print("INSTRUCTIONS:")
    print("="*70)
    print("1. Open 'roi_calibration.png' and check:")
    print("   - Are the BLUE GRID LINES aligned with table lines?")
    print("   - Are the QUESTION NUMBERS aligned with actual questions?")
    print("   - Are the A-E LABELS aligned with bubble columns?")
    print("   - Are the YELLOW BOXES centered on filled bubbles?")
    print("\n2. If NOT aligned, adjust these values in detect_answers.py:")
    print(f"   start_y = int(height * 0.22)  <- Move up: decrease, down: increase")
    print(f"   end_y = int(height * 0.95)    <- Expand: increase, shrink: decrease")
    print(f"   col1_start_x = int(width * 0.08)  <- Move left: decrease, right: increase")
    print(f"   col1_end_x = int(width * 0.48)")
    print(f"   col2_start_x = int(width * 0.52)")
    print(f"   col2_end_x = int(width * 0.92)")
    print("\n3. Run this script again after adjusting to verify")
    print("="*70)
    
    # Show interactive
    if show_interactive:
        # Resize for screen
        scale = 0.6
        display = cv2.resize(vis, None, fx=scale, fy=scale)
        cv2.imshow("ROI Calibration - Press any key to close", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def highlight_bubble(img, col_start_x, start_y, bubble_width, row_height, row, col, color):
    """Draw box around a specific bubble"""
    x1 = col_start_x + col * bubble_width + 5
    y1 = start_y + row * row_height + 5
    x2 = x1 + bubble_width - 10
    y2 = y1 + row_height - 10
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == "__main__":
    import sys
    
    # Check if warped image exists
    warped_file = "test_03_warped_raw.png"
    
    if len(sys.argv) > 1:
        warped_file = sys.argv[1]
    
    draw_roi_grid(warped_file, show_interactive=True)
"""
Generate 10 LJK (Lembar Jawaban Ujian) PNG files with filled answers.
Uses sample.jpg as the blank template and draws filled bubbles using OpenCV.

Strategy:
- Load sample.jpg (blank LJK with corner markers)
- Detect the paper boundary (largest white rectangle) from the photo
- Warp to a clean A4 canvas, then draw filled bubbles
- Save 10 variations with different randomized answers
"""

import sys
import io
# Fix Windows console encoding for special characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import json
import random
import os
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
TEMPLATE_PATH = "sample.jpg"
OUTPUT_DIR    = "ljk_filled"
NUM_SHEETS    = 10
SEED          = 42  # Reproducible randomness

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD & WARP TEMPLATE
# Strategy: detect the LJK paper boundary using contour detection
# on the photo (dark wooden background vs white paper = easy to separate)
# ============================================================
def order_points(pts):
    """Order points as TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_warped_template(img_path):
    """
    Load template photo and warp the LJK paper region to a clean A4 canvas.
    Works by finding the largest white rectangle (the paper) in the photo.
    """
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load template: {img_path}")

    h_orig, w_orig = image.shape[:2]
    print(f"      Original size: {w_orig}x{h_orig}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold: the paper is BRIGHT (white) on a DARK background
    # Use simple threshold — paper ~200+, dark wood ~50-100
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Morphological close to fill small holes in the paper region
    kernel = np.ones((25, 25), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"      Contours found: {len(contours)}")

    if not contours:
        raise RuntimeError("No contours found in template image.")

    # Pick the largest contour (should be the paper)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    print(f"      Largest contour area: {area:.0f} px^2 ({area / (w_orig * h_orig) * 100:.1f}% of image)")

    # Approximate to polygon
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    print(f"      Polygon vertices: {len(approx)}")

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
    else:
        # Fallback: use bounding rect if not a clean quad
        print("      Using bounding rect fallback (polygon not clean 4-sided)")
        x, y, w, h = cv2.boundingRect(largest)
        pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")

    rect = order_points(pts)
    dst = np.array([[0, 0], [1000, 0], [1000, 1414], [0, 1414]], dtype="float32")
    M_warp = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M_warp, (1000, 1414))
    return warped


# ============================================================
# BUBBLE GRID COORDINATES
# These are calibrated from detect_answers.py grid config.
# Canvas size after warp: 1000 x 1414
# ============================================================
CANVAS_W = 1000
CANVAS_H = 1414

# Y axis (from detect_answers.py)
START_Y_RATIO = 0.222
END_Y_RATIO   = 0.975

# Column X bounds (from detect_answers.py)
COL1_START_X_RATIO = 0.090
COL1_END_X_RATIO   = 0.350
COL2_START_X_RATIO = 0.590
COL2_END_X_RATIO   = 0.855

# 5 options per question (A=0 .. E=4) — equal width slices
BUBBLE_POSITIONS = [
    (0.00, 0.20),  # A
    (0.20, 0.40),  # B
    (0.40, 0.60),  # C
    (0.60, 0.80),  # D
    (0.80, 1.00),  # E
]

QUESTIONS_PER_COL = 15
TOTAL_QUESTIONS   = 30
OPTIONS           = ['A', 'B', 'C', 'D', 'E']

def get_bubble_center(question_no, answer_letter, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """
    Returns (x, y) pixel center of a bubble given question number (1-30)
    and answer letter (A-E).
    """
    col_idx   = 0 if question_no <= 15 else 1   # 0 = left col, 1 = right col
    row_in_col = (question_no - 1) % QUESTIONS_PER_COL  # 0-indexed row

    # Determine column X bounds
    if col_idx == 0:
        x_start_col = int(canvas_w * COL1_START_X_RATIO)
        x_end_col   = int(canvas_w * COL1_END_X_RATIO)
    else:
        x_start_col = int(canvas_w * COL2_START_X_RATIO)
        x_end_col   = int(canvas_w * COL2_END_X_RATIO)

    col_width = x_end_col - x_start_col

    # Y bounds
    start_y = int(canvas_h * START_Y_RATIO)
    end_y   = int(canvas_h * END_Y_RATIO)

    # Y steps for 15 rows
    y_steps = np.linspace(start_y, end_y, QUESTIONS_PER_COL + 1).astype(int)
    y_top    = y_steps[row_in_col]
    y_bottom = y_steps[row_in_col + 1]
    cy = (y_top + y_bottom) // 2

    # X position for answer
    opt_idx = OPTIONS.index(answer_letter)
    x_start_ratio, x_end_ratio = BUBBLE_POSITIONS[opt_idx]
    x_left  = x_start_col + int(col_width * x_start_ratio)
    x_right = x_start_col + int(col_width * x_end_ratio)
    cx = (x_left + x_right) // 2

    return (cx, cy)


def erase_bubble_region(canvas, question_no, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """
    Whiten out the entire row region for a question across all 5 options,
    effectively removing any pre-filled marks from the template.
    """
    col_idx   = 0 if question_no <= 15 else 1
    row_in_col = (question_no - 1) % QUESTIONS_PER_COL

    if col_idx == 0:
        x_start_col = int(canvas_w * COL1_START_X_RATIO) - 2
        x_end_col   = int(canvas_w * COL1_END_X_RATIO) + 2
    else:
        x_start_col = int(canvas_w * COL2_START_X_RATIO) - 2
        x_end_col   = int(canvas_w * COL2_END_X_RATIO) + 2

    start_y = int(canvas_h * START_Y_RATIO)
    end_y   = int(canvas_h * END_Y_RATIO)
    y_steps = np.linspace(start_y, end_y, QUESTIONS_PER_COL + 1).astype(int)

    overlap = 3
    y1 = max(0, y_steps[row_in_col] - overlap)
    y2 = min(canvas_h, y_steps[row_in_col + 1] + overlap)

    canvas[y1:y2, x_start_col:x_end_col] = 255  # Fill white


def draw_filled_bubble(canvas, cx, cy, radius=18):
    """Draw a solid dark filled circle (pencil-like) on the canvas."""
    # Main filled circle
    cv2.circle(canvas, (cx, cy), radius, (30, 30, 30), -1)
    # Slight texture: draw a slightly lighter inner ring to mimic pencil fill
    cv2.circle(canvas, (cx, cy), int(radius * 0.55), (20, 20, 20), -1)


def draw_empty_bubble(canvas, cx, cy, radius=18):
    """Draw an empty circle (ring only) on the canvas."""
    cv2.circle(canvas, (cx, cy), radius, (60, 60, 60), 2)


def generate_answers(strategy="random"):
    """
    Generate 30 answers.
    strategy: 'random' | 'all_correct' | 'mostly_correct'
    """
    return {i: random.choice(OPTIONS) for i in range(1, TOTAL_QUESTIONS + 1)}


# ============================================================
# MAIN GENERATION LOOP
# ============================================================
def main():
    print("=" * 55)
    print("  LJK Filled Sheet Generator")
    print("=" * 55)

    # Load & warp color template
    print(f"\n[1/3] Loading template: {TEMPLATE_PATH}")
    template = get_warped_template(TEMPLATE_PATH)
    if template is None:
        print("❌ Error: Gagal warp template. Periksa sample.jpg.")
        return

    print(f"      Template size: {template.shape[1]}x{template.shape[0]} px")

    # Convert template to grayscale working copy base
    # (keep it as BGR for final output, but work with a clean base)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # Use white-ish background
    _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)

    print(f"\n[2/3] Generating {NUM_SHEETS} filled sheets...")
    print(f"      Randomization seed: {SEED}")

    # Define varied answer scenarios for realistic testing
    # Some students answer all, some leave blanks, some are mostly correct vs key
    answer_key = {
        1:"A", 2:"D", 3:"B", 4:"C", 5:"E",
        6:"B", 7:"A", 8:"C", 9:"D", 10:"E",
        11:"C", 12:"A", 13:"B", 14:"D", 15:"E",
        16:"A", 17:"C", 18:"B", 19:"D", 20:"E",
        21:"B", 22:"A", 23:"C", 24:"D", 25:"E",
        26:"A", 27:"B", 28:"C", 29:"D", 30:"E"
    }

    # Load from file if exists
    if os.path.exists("answer_key.json"):
        with open("answer_key.json") as f:
            raw = json.load(f)
        answer_key = {int(k): v for k, v in raw.items()}

    # Scenario descriptions for each of the 10 sheets
    scenarios = [
        {"name": "siswa_01_sempurna",       "correct_pct": 1.00, "blank_pct": 0.00, "double_pct": 0.00},
        {"name": "siswa_02_bagus",          "correct_pct": 0.87, "blank_pct": 0.00, "double_pct": 0.00},
        {"name": "siswa_03_sedang",         "correct_pct": 0.70, "blank_pct": 0.07, "double_pct": 0.00},
        {"name": "siswa_04_banyak_kosong",  "correct_pct": 0.50, "blank_pct": 0.20, "double_pct": 0.00},
        {"name": "siswa_05_double_bubble",  "correct_pct": 0.70, "blank_pct": 0.00, "double_pct": 0.10},
        {"name": "siswa_06_acak",           "correct_pct": 0.40, "blank_pct": 0.10, "double_pct": 0.03},
        {"name": "siswa_07_hampir_penuh",   "correct_pct": 0.93, "blank_pct": 0.03, "double_pct": 0.00},
        {"name": "siswa_08_banyak_salah",   "correct_pct": 0.30, "blank_pct": 0.00, "double_pct": 0.00},
        {"name": "siswa_09_mix",            "correct_pct": 0.60, "blank_pct": 0.10, "double_pct": 0.07},
        {"name": "siswa_10_cukup",          "correct_pct": 0.77, "blank_pct": 0.03, "double_pct": 0.00},
    ]

    all_answers_log = {}

    for sheet_idx, scenario in enumerate(scenarios, start=1):
        name         = scenario["name"]
        correct_pct  = scenario["correct_pct"]
        blank_pct    = scenario["blank_pct"]
        double_pct   = scenario["double_pct"]

        # Build answer set for this sheet
        sheet_answers = {}  # {q_no: answer_letter | None | 'DOUBLE'}
        questions = list(range(1, TOTAL_QUESTIONS + 1))
        random.shuffle(questions)

        n_blank  = int(TOTAL_QUESTIONS * blank_pct)
        n_double = int(TOTAL_QUESTIONS * double_pct)
        n_correct = int(TOTAL_QUESTIONS * correct_pct)

        # Assign roles
        blank_qs  = set(questions[:n_blank])
        double_qs = set(questions[n_blank:n_blank + n_double])
        correct_qs = set(questions[n_blank + n_double : n_blank + n_double + n_correct])
        wrong_qs   = set(questions) - blank_qs - double_qs - correct_qs

        for q in range(1, TOTAL_QUESTIONS + 1):
            if q in blank_qs:
                sheet_answers[q] = None
            elif q in double_qs:
                # Pick 2 different answers
                opts = OPTIONS.copy()
                chosen = random.sample(opts, 2)
                sheet_answers[q] = chosen  # List = double bubble
            elif q in correct_qs:
                sheet_answers[q] = answer_key.get(q, random.choice(OPTIONS))
            else:  # wrong
                correct = answer_key.get(q, 'A')
                wrong_opts = [o for o in OPTIONS if o != correct]
                sheet_answers[q] = random.choice(wrong_opts)

        # ---- DRAW ON CANVAS ----
        # Start from a clean copy of the template (converted to BGR)
        canvas = cv2.cvtColor(template_binary, cv2.COLOR_GRAY2BGR)

        # First, erase all bubble areas to get a clean slate
        for q in range(1, TOTAL_QUESTIONS + 1):
            erase_bubble_region(canvas, q)

        # Draw empty circles for all options
        for q in range(1, TOTAL_QUESTIONS + 1):
            for opt in OPTIONS:
                cx, cy = get_bubble_center(q, opt)
                draw_empty_bubble(canvas, cx, cy)

        # Draw filled bubbles
        for q, ans in sheet_answers.items():
            if ans is None:
                continue  # blank — leave as empty circle
            elif isinstance(ans, list):
                # Double bubble
                for a in ans:
                    cx, cy = get_bubble_center(q, a)
                    draw_filled_bubble(canvas, cx, cy)
            else:
                # Single answer
                cx, cy = get_bubble_center(q, ans)
                draw_filled_bubble(canvas, cx, cy)

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        cv2.imwrite(out_path, canvas)

        # Log
        single_answers = {
            q: (ans if not isinstance(ans, list) else "DOUBLE")
            for q, ans in sheet_answers.items()
        }
        all_answers_log[name] = single_answers

        # Quick stats
        n_b = sum(1 for v in sheet_answers.values() if v is None)
        n_d = sum(1 for v in sheet_answers.values() if isinstance(v, list))
        n_filled = TOTAL_QUESTIONS - n_b - n_d
        print(f"  [{sheet_idx:2d}/10] {name}.png  (isi={n_filled}, kosong={n_b}, double={n_d})")

    # Save answers log for verification
    log_path = os.path.join(OUTPUT_DIR, "answers_log.json")
    with open(log_path, "w") as f:
        json.dump(all_answers_log, f, indent=2)

    print(f"\n[3/3] Selesai!")
    print(f"      Output folder : {os.path.abspath(OUTPUT_DIR)}/")
    print(f"      Answers log   : {log_path}")
    print(f"\n  Tip: Cek 'answers_log.json' untuk verifikasi jawaban setiap sheet.")
    print("=" * 55)


if __name__ == "__main__":
    main()

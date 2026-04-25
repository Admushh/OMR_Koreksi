"""
QUICK OMR TEST SCRIPT  (updated for v2 pipeline)
Usage: python quick_test.py [image_file]
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import json
import numpy as np

from omr_core.preprocess import preprocess_image, preprocess_for_answers, preprocess_for_markers
from omr_core.detect_sheet import find_paper
from omr_core.detect_answers import detect_answers
from omr_core.grading import grade_answers

# ──────────────────────────────────────────────────────────
# ANSWER KEY — edit to match your sheet
# ──────────────────────────────────────────────────────────
ANSWER_KEY = {
    1: 'A',  2: 'D',  3: 'B',  4: 'D',  5: 'E',
    6: 'B',  7: 'E',  8: 'B',  9: 'D',  10: 'B',
    11: 'A', 12: 'C', 13: 'D', 14: 'E', 15: 'B',
    16: 'B', 17: 'C', 18: 'B', 19: 'D', 20: 'B',
    21: 'D', 22: 'E', 23: 'A', 24: 'C', 25: 'B',
    26: 'D', 27: 'B', 28: 'D', 29: 'A', 30: 'E',
}

INPUT_IMAGE = sys.argv[1] if len(sys.argv) > 1 else "sample 2.png"

# ==========================================================

def sep(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def test_omr():
    sep("OMR QUICK TEST  v2")

    # Load
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"\n  ERROR: Cannot open '{INPUT_IMAGE}'")
        return False
    print(f"\n  Image : {INPUT_IMAGE}  ({img.shape[1]}x{img.shape[0]} px)")

    # ── Step 1: preprocess for marker detection ────────────────────────────
    sep("STEP 1: Preprocessing (corner markers)")
    thresh = preprocess_image(img)
    cv2.imwrite("test_01_preprocessed.png", thresh)
    print("  Saved: test_01_preprocessed.png")

    # ── Step 2: detect markers ─────────────────────────────────────────────
    sep("STEP 2: Marker detection + perspective warp")
    debug_img = img.copy()
    result = find_paper(thresh, debug_image=debug_img)

    cv2.imwrite("test_02_marker_detection.png", debug_img)
    print("  Saved: test_02_marker_detection.png")

    # Retry with padding if needed
    if result is None:
        print("  [WARN] First attempt failed, retrying with padding...")
        PAD = 50
        padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])
        thresh_p = preprocess_image(padded)
        debug_p  = padded.copy()
        result   = find_paper(thresh_p, debug_image=debug_p)
        cv2.imwrite("test_02_marker_detection.png", debug_p)
        if result is None:
            print("  ERROR: Markers not found even after padding.")
            print("  -> Check test_02_marker_detection.png")
            return False
        # Use padded for the rest
        img = padded

    warped_thresh, M_warp = result
    cv2.imwrite("test_03_warped_thresh.png", warped_thresh)
    print("  Saved: test_03_warped_thresh.png")

    # ── Step 3: warp ORIGINAL grayscale ───────────────────────────────────
    sep("STEP 3: Warp original gray image")
    src_gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.warpPerspective(src_gray, M_warp, (1000, 1414))
    cv2.imwrite("test_03_warped.png", warped_gray)
    print("  Saved: test_03_warped.png  (this is what answer detection uses)")

    # ── Step 4: preprocess for answers ────────────────────────────────────
    sep("STEP 4: Preprocess for bubble detection")
    warped_ready = preprocess_for_answers(warped_gray)
    cv2.imwrite("test_04_warped_ready.png", warped_ready)
    print("  Saved: test_04_warped_ready.png")

    # ── Step 5: detect answers ─────────────────────────────────────────────
    sep("STEP 5: Bubble detection")
    answers = detect_answers(warped_ready, num_questions=30, debug=True)
    # debug=True saves debug_grid_overlay.png

    # ── Step 6: grade ──────────────────────────────────────────────────────
    sep("STEP 6: Grading")
    result_grade = grade_answers(answers, ANSWER_KEY)

    # ── Results ────────────────────────────────────────────────────────────
    sep("RESULTS")
    score   = result_grade["score"]
    summary = result_grade["summary"]
    details = result_grade["details"]

    print(f"\n  SCORE   : {score:.1f} / 100")
    print(f"  Correct : {summary['correct']} / {summary['total']}")
    print(f"  Wrong   : {summary['wrong']}")
    print(f"  Empty   : {summary['empty']}")
    print(f"  Double  : {summary['double']}")

    wrong_empty = [(q, d) for q, d in details.items()
                   if d["status"] != "CORRECT"]
    if wrong_empty:
        print(f"\n  Problems:")
        for q, d in sorted(wrong_empty, key=lambda x: x[0]):
            ans = d["student"] or "(kosong)"
            print(f"    Q{q:>2}: detected={ans}  correct={d['correct']}  [{d['status']}]")
    else:
        print("\n  PERFECT! All answers correct.")

    # Save JSON
    with open("test_result.json", "w") as f:
        json.dump(result_grade, f, indent=2)
    print("\n  Saved: test_result.json")

    sep("DEBUG FILES")
    print("  test_01_preprocessed.png   — corner binary (marker detection input)")
    print("  test_02_marker_detection.png — green boxes on detected markers")
    print("  test_03_warped.png          — warped GRAY image (answer detection input)")
    print("  test_04_warped_ready.png    — after CLAHE preprocess")
    print("  debug_grid_overlay.png     — grid + scored cells overlay")

    return True


if __name__ == "__main__":
    ok = test_omr()
    sep()
    print("  " + ("TEST PASSED" if ok else "TEST FAILED"))
    sep()

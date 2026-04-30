
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import json
import time
import os

from omr_core.preprocess import preprocess_for_markers, preprocess_for_answers
from omr_core.detect_sheet import find_paper
from omr_core.detect_answers import detect_answers
from omr_core.grading import grade_answers

# ══════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════

# Expected answers based on visual inspection of each image
EXPECTED_ANSWERS = {
    "Kunjab.jpg": {
        1:'A', 2:'D', 3:'B', 4:'D', 5:'E',
        6:'B', 7:'E', 8:'B', 9:'D', 10:'B',
        11:'A', 12:'C', 13:'D', 14:'E', 15:'B',
        16:'B', 17:'C', 18:'B', 19:'D', 20:'B',
        21:'D', 22:'E', 23:'A', 24:'C', 25:'B',
        26:'D', 27:'B', 28:'D', 29:'A', 30:'E',
    },
    "sample.png": {
        1:'A', 2:'B', 3:'C', 4:'D', 5:'E',
        6:'A', 7:'B', 8:'C', 9:'D', 10:'E',
        11:'A', 12:'B', 13:'C', 14:'D', 15:'E',
        16:'A', 17:'B', 18:'C', 19:'D', 20:'E',
        21:'A', 22:'B', 23:'C', 24:'D', 25:'E',
        26:'A', 27:'B', 28:'C', 29:'D', 30:'E',
    },
}

TEST_FILES = [
    ("Kunjab.jpg",    "Clean scan (kondisi ideal)"),
    ("sample.png",  "Noisy (shadow tangan + tulisan tangan + miring ringan)"),
]

OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def sep(title="", char="=", width=65):
    print(f"\n{char * width}")
    if title:
        print(f"  {title}")
        print(f"{char * width}")


def process_image(fname, label):
    """
    Run the full pipeline on one image.
    Returns dict with timing, answers, and accuracy info.
    """
    result = {
        "file": fname,
        "label": label,
        "timings": {},
        "detected_answers": None,
        "accuracy": None,
        "errors": [],
    }

    # Load image
    img = cv2.imread(fname)
    if img is None:
        result["errors"].append(f"Cannot open file: {fname}")
        return result
    h_orig, w_orig = img.shape[:2]
    print(f"  Image: {fname} ({w_orig}x{h_orig} px)")

    # ── STAGE 1: Preprocess for markers ─────────────────────
    t0 = time.perf_counter()
    thresh = preprocess_for_markers(img)
    t1 = time.perf_counter()
    result["timings"]["preprocess_markers"] = round((t1 - t0) * 1000, 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname.split('.')[0]}_01_thresh.png"), thresh)

    # ── STAGE 2: Detect markers + warp ──────────────────────
    t0 = time.perf_counter()
    debug_img = img.copy()
    detect_result = find_paper(thresh, debug_image=debug_img)
    t1 = time.perf_counter()
    result["timings"]["find_paper"] = round((t1 - t0) * 1000, 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname.split('.')[0]}_02_markers.png"), debug_img)

    # Retry with padding if needed
    used_padding = False
    src_img = img
    if detect_result is None:
        print("  [RETRY] Adding padding and retrying...")
        PAD = 50
        padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])
        thresh_p = preprocess_for_markers(padded)
        debug_p = padded.copy()

        t0 = time.perf_counter()
        detect_result = find_paper(thresh_p, debug_image=debug_p)
        t1 = time.perf_counter()
        result["timings"]["find_paper_retry"] = round((t1 - t0) * 1000, 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname.split('.')[0]}_02_markers_padded.png"), debug_p)

        if detect_result is None:
            result["errors"].append("Markers not found even after padding")
            return result
        used_padding = True
        src_img = padded

    warped_binary, M_warp = detect_result
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname.split('.')[0]}_03_warped_binary.png"), warped_binary)

    # ── STAGE 3: Warp original grayscale ────────────────────
    t0 = time.perf_counter()
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.warpPerspective(src_gray, M_warp, (1000, 1414))
    t1 = time.perf_counter()
    result["timings"]["warp_gray"] = round((t1 - t0) * 1000, 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname.split('.')[0]}_04_warped_gray.png"), warped_gray)

    # ── STAGE 4: Preprocess for answers ─────────────────────
    t0 = time.perf_counter()
    warped_ready = preprocess_for_answers(warped_gray)
    t1 = time.perf_counter()
    result["timings"]["preprocess_answers"] = round((t1 - t0) * 1000, 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname.split('.')[0]}_05_warped_enhanced.png"), warped_ready)

    # ── STAGE 5: Detect answers ─────────────────────────────
    t0 = time.perf_counter()
    detected = detect_answers(warped_ready, num_questions=30, debug=True)
    t1 = time.perf_counter()
    result["timings"]["detect_answers"] = round((t1 - t0) * 1000, 1)

    result["detected_answers"] = detected

    # Move debug overlay to output dir
    debug_overlay_src = "debug_grid_overlay.png"
    debug_overlay_dst = os.path.join(OUTPUT_DIR, f"{fname.split('.')[0]}_06_grid_overlay.png")
    if os.path.exists(debug_overlay_src):
        import shutil
        shutil.move(debug_overlay_src, debug_overlay_dst)

    # ── STAGE 6: Accuracy check ─────────────────────────────
    expected = EXPECTED_ANSWERS.get(fname)
    if expected:
        correct = 0
        wrong = 0
        empty = 0
        double = 0
        wrong_details = []

        for q in range(1, 31):
            det = detected.get(q)
            exp = expected.get(q)

            if det == "DOUBLE":
                double += 1
                wrong_details.append((q, det, exp))
            elif det is None:
                empty += 1
                wrong_details.append((q, "(kosong)", exp))
            elif det == exp:
                correct += 1
            else:
                wrong += 1
                wrong_details.append((q, det, exp))

        accuracy = (correct / 30) * 100
        result["accuracy"] = {
            "correct": correct,
            "wrong": wrong,
            "empty": empty,
            "double": double,
            "total": 30,
            "pct": round(accuracy, 1),
            "problems": wrong_details,
        }

    result["used_padding"] = used_padding
    return result


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    sep("OMR IMAGE PROCESSING BENCHMARK  v3-robust")

    all_results = []

    for fname, label in TEST_FILES:
        sep(f"TEST: {fname}")
        print(f"  Description: {label}")

        res = process_image(fname, label)
        all_results.append(res)

        # Print errors
        if res["errors"]:
            for e in res["errors"]:
                print(f"  ❌ ERROR: {e}")
            continue

        # Print timings
        sep("Timings", char="-")
        total_ms = 0
        for stage, ms in res["timings"].items():
            print(f"    {stage:<25} : {ms:>8.1f} ms")
            total_ms += ms
        print(f"    {'TOTAL':<25} : {total_ms:>8.1f} ms")

        # Print accuracy
        acc = res.get("accuracy")
        if acc:
            sep("Accuracy", char="-")
            print(f"    Score    : {acc['pct']:.1f}%  ({acc['correct']}/30 benar)")
            print(f"    Wrong    : {acc['wrong']}")
            print(f"    Empty    : {acc['empty']}")
            print(f"    Double   : {acc['double']}")

            if acc["problems"]:
                print(f"\n    Masalah ({len(acc['problems'])} soal):")
                for q, det, exp in acc["problems"]:
                    print(f"      Q{q:>2}: detected={str(det):<8} expected={exp}")
            else:
                print("\n    🎉 PERFECT! Semua jawaban terdeteksi benar!")

    # ── SUMMARY TABLE ──────────────────────────────────────────
    sep("RINGKASAN PERBANDINGAN")
    print(f"  {'File':<20} | {'Akurasi':>8} | {'Benar':>5} | {'Salah':>5} | {'Kosong':>6} | {'Double':>6} | {'Total ms':>10} | {'Padding':>7}")
    print(f"  {'-'*90}")
    for res in all_results:
        if res["errors"]:
            print(f"  {res['file']:<20} | {'ERROR':>8} |       |       |        |        |            |")
            continue
        acc = res.get("accuracy", {})
        total_ms = sum(res["timings"].values())
        pad_str = "Yes" if res.get("used_padding") else "No"
        print(f"  {res['file']:<20} | {acc.get('pct', 0):>7.1f}% | "
              f"{acc.get('correct', 0):>5} | {acc.get('wrong', 0):>5} | "
              f"{acc.get('empty', 0):>6} | {acc.get('double', 0):>6} | "
              f"{total_ms:>9.1f}ms | {pad_str:>7}")
    sep()

    # ── SAVE JSON ──────────────────────────────────────────────
    json_out = {}
    for res in all_results:
        json_out[res["file"]] = {
            "timings": res.get("timings", {}),
            "accuracy": res.get("accuracy", {}),
            "detected_answers": {str(k): v for k, v in res.get("detected_answers", {}).items()} if res.get("detected_answers") else None,
            "errors": res.get("errors", []),
            "used_padding": res.get("used_padding", False),
        }
    json_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  Saved: {json_path}")
    print(f"  Debug images: {OUTPUT_DIR}/")
    sep()


if __name__ == "__main__":
    main()

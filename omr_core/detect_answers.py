"""
detect_answers.py — Robust LJK Answer Detection (v3-robust)
============================================================

Detects student answers (A, B, C, D, E) from a warped, preprocessed
grayscale image.

Key improvements over v2:
- Uses the enhanced grayscale image for scoring, NOT binary.
- Bubble score is based on mean pixel intensity instead of positive pixel count.
- Z-score based statistical detection instead of global absolute thresholds.
  (This naturally adapts to local shadows or lighting gradients per row).
- Detects multiple filled answers as "DOUBLE" if they have similar scores.
"""

import cv2
import numpy as np


# Option constants
OPTIONS = ['A', 'B', 'C', 'D', 'E']


def _score_bubble(cell_img):
    """
    Score a bubble based on its darkness.
    Input: cell image (grayscale, already cropped to the bubble area)
    
    Returns: float score (higher = darker/filled).
    Formula: 255 - mean_intensity
    
    Why this is better than binary pixel count:
    A lightly shaded bubble might not cross a binary threshold, but its
    overall average intensity will still be markedly lower (darker) than a blank bubble.
    """
    if cell_img.size == 0:
        return 0.0

    mean_val = np.mean(cell_img)
    # Invert so that black (0) = 255 score, white (255) = 0 score
    return 255.0 - mean_val


def _read_column(working_gray, x_start, x_end, y_start, y_end,
                 n_questions, debug_img=None):
    """
    Scan n_questions rows in a column region.
    Returns list of 'A'–'E', 'DOUBLE', or None per question.
    """
    col_w = x_end - x_start
    col_h = y_end - y_start
    roi = working_gray[y_start:y_end, x_start:x_end]
    y_cuts = np.linspace(0, col_h, n_questions + 1, dtype=int)

    # Standard bounds for 5 options evenly spaced
    OPTION_BOUNDS = [
        (0.00, 0.20),  # Option A
        (0.20, 0.40),  # Option B
        (0.40, 0.60),  # Option C
        (0.60, 0.80),  # Option D
        (0.80, 1.00),  # Option E
    ]

    answers = []

    for q_idx in range(n_questions):
        yt = y_cuts[q_idx]
        yb = y_cuts[q_idx + 1]

        # Score each of 5 options
        scores = []
        for (xs_r, xe_r) in OPTION_BOUNDS:
            xt = int(col_w * xs_r)
            xb = int(col_w * xe_r)
            inset = 4  # Avoid cell borders
            cell = roi[yt + inset: yb - inset, xt + inset: xb - inset]
            scores.append(_score_bubble(cell))

        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # ── Decision: Z-score based (robust against shadows) ───
        # Instead of ratio to median (fails with shadow gradient),
        # use statistical outlier detection:
        #   - If a score is > mean + Z_THRESH × stdev, it's an outlier = filled
        #   - This works because in a row with 1 filled + 4 empty,
        #     the filled one is a clear statistical outlier
        #   - Shadows inflate ALL scores equally per row, so the
        #     relative difference (z-score) remains stable
        #
        # Z_THRESH=1.0: filled bubbles typically 1.2-2.5 stdev above mean
        # MIN_ABS_DIFF: absolute score gap to avoid noise triggers
        Z_THRESH = 1.0
        MIN_ABS_DIFF = 5.0

        if std_score < 2.0:
            # Very uniform scores = all empty (no filled bubble)
            filled = []
        else:
            threshold = mean_score + Z_THRESH * std_score
            filled = [
                i for i, s in enumerate(scores)
                if s >= threshold and (s - mean_score) >= MIN_ABS_DIFF
            ]

        if len(filled) == 0:
            answer = None
        elif len(filled) == 1:
            answer = OPTIONS[filled[0]]
        else:
            # Multiple filled — check if truly double or just one dominant
            top_idx = scores.index(max_score)
            scores_sorted = sorted(scores)
            second_max = scores_sorted[-2]

            # Genuine double: runner-up ≥ 85% of winner AND both above threshold
            if max_score > 0 and second_max >= max_score * 0.85:
                answer = "DOUBLE"
            else:
                answer = OPTIONS[top_idx]

        answers.append(answer)

        # ── Debug overlay ───────────────────────────────────────
        if debug_img is not None:
            for opt_idx, (xs_r, xe_r) in enumerate(OPTION_BOUNDS):
                gx1 = x_start + int(col_w * xs_r) + 4
                gx2 = x_start + int(col_w * xe_r) - 4
                gy1 = y_start + yt + 4
                gy2 = y_start + yb - 4
                
                # Pick color
                if answer == "DOUBLE" and opt_idx in filled:
                    color = (0, 0, 255) # Red for double
                elif answer == OPTIONS[opt_idx]:
                    color = (0, 255, 0) # Green for chosen
                else:
                    color = (255, 255, 0) # Cyan for scanned options

                cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), color, 1)

                # Overlay score value
                score_str = f"{scores[opt_idx]:.0f}"
                cv2.putText(debug_img, score_str, (gx1 + 2, gy2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return answers


def detect_answers(warped_ready, num_questions=30, debug=False):
    """
    Read bubbles arranged in 2 columns (15 questions each).
    
    Parameters
    ----------
    warped_ready : grayscale image from preprocess_for_answers()
    num_questions: Total number of questions expected (max 30)
    
    Returns
    -------
    dict: { 1: 'A', 2: 'C', 3: None, 4: 'DOUBLE', ... }
    """
    h, w = warped_ready.shape[:2]
    
    if debug:
        # Create a BGR copy for colored overlay
        debug_img = cv2.cvtColor(warped_ready, cv2.COLOR_GRAY2BGR)
        print(f"[detect_answers] Canvas: {w}x{h}  Questions: {num_questions}")
    else:
        debug_img = None

    # Geometry relative to 1000x1414 canonical canvas
    y_start = int(h * 0.222)
    y_end   = int(h * 0.975)

    c1_xs = int(w * 0.090)
    c1_xe = int(w * 0.350)

    c2_xs = int(w * 0.590)
    c2_xe = int(w * 0.855)

    all_answers = {}
    
    # Process Column 1
    q_col1 = min(num_questions, 15)
    col1_ans = _read_column(warped_ready, c1_xs, c1_xe, y_start, y_end, q_col1, debug_img)
    for i, a in enumerate(col1_ans):
        all_answers[i + 1] = a

    # Process Column 2
    if num_questions > 15:
        q_col2 = min(num_questions - 15, 15)
        col2_ans = _read_column(warped_ready, c2_xs, c2_xe, y_start, y_end, q_col2, debug_img)
        for i, a in enumerate(col2_ans):
            all_answers[15 + i + 1] = a

    if debug:
        cv2.imwrite("debug_grid_overlay.png", debug_img)
        print("[detect_answers] Saved: debug_grid_overlay.png")

        # Basic text log
        print("\n=============================================")
        print("  REKAP HASIL DETEKSI")
        print("=============================================")
        print("  No    | Jawaban")
        print("  -------------------------")
        for i in range(1, num_questions + 1):
            val = all_answers.get(i)
            if val == "DOUBLE":
                print(f"  Q{i:<4} | [DOUBLE BUBBLE]")
            elif val is None:
                print(f"  Q{i:<4} | [KOSONG]")
            else:
                print(f"  Q{i:<4} | {val}")
        print("=============================================\n")

    return all_answers
import cv2
import numpy as np


# Option constants
OPTIONS = ['A', 'B', 'C', 'D', 'E']

# Precise Y centers of the 15 questions in the 1000x1414 canonical canvas
ROW_Y_CENTERS = [
    320, 389, 459, 529, 599, 669, 739, 810, 880, 951, 1022, 1093, 1164, 1236, 1307
]


def _score_bubble(cell_img):

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
        yc = ROW_Y_CENTERS[q_idx]
        yt = yc - 35
        yb = yc + 35

        # Score each of 5 options
        scores = []
        for (xs_r, xe_r) in OPTION_BOUNDS:
            xt = x_start + int(col_w * xs_r)
            xb = x_start + int(col_w * xe_r)
            inset = 8  # Avoid cell borders
            cell = working_gray[yt + inset: yb - inset, xt + inset: xb - inset]
            scores.append(_score_bubble(cell))

        max_score = max(scores) if scores else 0
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Tuned thresholds to avoid false positives
        Z_THRESH = 1.2
        MIN_ABS_DIFF = 10.0
        MIN_STD_SCORE = 5.0

        if std_score < MIN_STD_SCORE:
            # Very uniform scores = all empty (no filled bubble)
            filled = []
        else:
            threshold = mean_score + Z_THRESH * std_score
            filled = [
                i for i, s in enumerate(scores)
                if s >= threshold and (s - mean_score) >= MIN_ABS_DIFF
            ]

        # Initialize second_max for debug overlay
        scores_sorted = sorted(scores)
        second_max = scores_sorted[-2]

        if len(filled) == 0:
            answer = None
        else:
            top_idx = scores.index(max_score)
            # Check for DOUBLE bubble: if second max is close to max and significantly above empty baseline
            mean_other = np.mean(scores_sorted[:-2])
            if second_max >= max_score * 0.85 and (second_max - mean_other) > 10.0:
                answer = "DOUBLE"
            elif len(filled) == 1:
                answer = OPTIONS[filled[0]]
            else:
                answer = OPTIONS[top_idx]

        answers.append(answer)

        # ── Debug overlay ───────────────────────────────────────
        if debug_img is not None:
            for opt_idx, (xs_r, xe_r) in enumerate(OPTION_BOUNDS):
                gx1 = x_start + int(col_w * xs_r) + 4
                gx2 = x_start + int(col_w * xe_r) - 4
                gy1 = yt + 4
                gy2 = yb - 4
                
                # Pick color
                if answer == "DOUBLE" and scores[opt_idx] >= second_max:
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

    h, w = warped_ready.shape[:2]
    
    if debug:
        # Create a BGR copy for colored overlay
        debug_img = cv2.cvtColor(warped_ready, cv2.COLOR_GRAY2BGR)
        print(f"[detect_answers] Canvas: {w}x{h}  Questions: {num_questions}")
    else:
        debug_img = None

    # Geometry relative to 1000x1414 canonical canvas
    y_start = 285
    y_end   = 1342

    c1_xs = int(w * 0.090)  # 90
    c1_xe = int(w * 0.350)  # 350

    c2_xs = 594
    c2_xe = 854

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
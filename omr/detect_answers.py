import cv2
import numpy as np

def detect_answers(thresh, total_questions=30, total_choices=5):
    answers = []

    h, w = thresh.shape
    box_h = h // total_questions
    box_w = w // total_choices

    for q in range(total_questions):
        row = thresh[q * box_h:(q + 1) * box_h, :]
        bubbled = None
        max_pixels = 0

        for c in range(total_choices):
            col = row[:, c * box_w:(c + 1) * box_w]
            total = cv2.countNonZero(col)

            if total > max_pixels:
                max_pixels = total
                bubbled = c

        if bubbled is None:
            answers.append("-")
        else:
            answers.append(chr(65 + bubbled))

    return answers
    

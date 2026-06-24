import cv2
import numpy as np
import sys
import os

from omr_core.ocr import extract_name_and_id

def run_ocr_test():
    # Load reference warped sheet using absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    warped_img_path = os.path.join(script_dir, "scratch_revisi_warped.png")
    
    if not os.path.exists(warped_img_path):
        print(f"Error: {warped_img_path} not found in workspace.")
        sys.exit(1)

    img = cv2.imread(warped_img_path)
    if img is None:
        print("Error: Cannot read warped image.")
        sys.exit(1)

    simulated = img.copy()

    # Simulate Name: "ADIMAS" (Y center ~204)
    name_chars = ["A", "D", "I", "M", "A", "S"]
    name_centers = [272, 312, 352, 390, 430, 472]
    for char, cx in zip(name_chars, name_centers):
        cv2.putText(simulated, char, (cx - 7, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    # Simulate ID: "12345678" (Y center ~234)
    id_chars = ["1", "2", "3", "4", "5", "6", "7", "8"]
    id_centers = [272, 312, 352, 390, 430, 472, 512, 552]
    for char, cx in zip(id_chars, id_centers):
        cv2.putText(simulated, char, (cx - 7, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    # Convert simulated to grayscale
    gray_simulated = cv2.cvtColor(simulated, cv2.COLOR_BGR2GRAY)

    # Execute extraction
    print("Running Name & ID extraction pipeline...")
    name_text, id_text = extract_name_and_id(gray_simulated)

    print("\n=============================================")
    print("  OCR VALIDATION RESULT")
    print("=============================================")
    print(f"  Extracted Name: '{name_text}' (Expected: 'ADIMAS')")
    print(f"  Extracted ID  : '{id_text}' (Expected: '12345678')")
    print("=============================================\n")

    # Cleaned outputs are written to the script directory:
    cleaned_name_path = os.path.join(script_dir, "scratch_ocr_cleaned_name.png")
    cleaned_id_path = os.path.join(script_dir, "scratch_ocr_cleaned_id.png")
    print("Preprocessed crops saved to:")
    print(f"  - {cleaned_name_path}")
    print(f"  - {cleaned_id_path}")

    # Allow spaces in name comparison (e.g. "ADIM A S" is fine as long as chars are correct)
    if name_text.replace(" ", "") == "ADIMAS" and id_text == "12345678":
        print("SUCCESS: OCR test passed!")
    else:
        print("FAIL: OCR mismatch!")
        sys.exit(1)

if __name__ == "__main__":
    run_ocr_test()

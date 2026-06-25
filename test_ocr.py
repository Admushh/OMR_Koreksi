import cv2
import numpy as np
import sys
import os

from omr_core.ocr import extract_name_and_id

def run_ocr_test():
    # Load reference warped sheet using absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    warped_img_path = os.path.join(script_dir, "IMG_3345.png")
    
    if not os.path.exists(warped_img_path):
        print(f"Error: {warped_img_path} not found in workspace.")
        sys.exit(1)

    img = cv2.imread(warped_img_path)
    if img is None:
        print("Error: Cannot read warped image.")
        sys.exit(1)

    # Convert loaded image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Execute extraction directly on the raw image
    print("Running Name & ID extraction pipeline on raw IMG_3344.PNG...")
    name_text, id_text = extract_name_and_id(gray_img)

    print("\n=============================================")
    print("  RAW OCR EXTRACTION RESULT")
    print("=============================================")
    print(f"  Extracted Name: '{name_text}'")
    print(f"  Extracted ID  : '{id_text}'")
    print("=============================================\n")

    # Cleaned outputs are written to the script directory:
    cleaned_name_path = os.path.join(script_dir, "scratch_ocr_cleaned_name.png")
    cleaned_id_path = os.path.join(script_dir, "scratch_ocr_cleaned_id.png")
    print("Preprocessed crops saved to:")
    print(f"  - {cleaned_name_path}")
    print(f"  - {cleaned_id_path}")

if __name__ == "__main__":
    run_ocr_test()

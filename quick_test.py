"""
QUICK OMR TEST SCRIPT
=====================
Test your OMR system without building any app!

Instructions:
1. Fill in your answer key below
2. Fill out a physical OMR sheet with a pen
3. Take a photo or scan it (save as 'test_sheet.png')
4. Run this script: python quick_test.py
5. Check the results and debug images
"""

import cv2
import sys
import json
from omr_core.preprocess import preprocess_for_detection, preprocess_for_grading
from omr_core.detect_sheet import find_paper
from omr_core.detect_answers import detect_answers, grade_answers

# ============================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================

# Your answer key (edit this with correct answers)
ANSWER_KEY = {
    1: 'A',  2: 'B',  3: 'C',  4: 'D',  5: 'E',
    6: 'B',  7: 'C',  8: 'D',  9: 'E',  10: 'A',
    11: 'C', 12: 'D', 13: 'E', 14: 'A', 15: 'B',
    16: 'D', 17: 'E', 18: 'A', 19: 'B', 20: 'C',
    21: 'E', 22: 'A', 23: 'B', 24: 'C', 25: 'D',
    26: 'A', 27: 'B', 28: 'C', 29: 'D', 30: 'E',
}

# Input image filename (your scanned/photo OMR sheet)
INPUT_IMAGE = "test_scanned.jpg"  # Change this to your file

# ============================================================
# TEST SCRIPT - DON'T EDIT BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
# ============================================================

def test_omr():
    print("="*80)
    print(" OMR SYSTEM - QUICK TEST ".center(80, "="))
    print("="*80)
    
    # Check if file exists
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"\n❌ ERROR: File '{INPUT_IMAGE}' not found!")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure your OMR sheet image is in the same folder")
        print(f"  2. Rename it to '{INPUT_IMAGE}' or change INPUT_IMAGE in this script")
        print(f"  3. Supported formats: PNG, JPG, JPEG")
        return False
    
    print(f"\n✅ Image loaded: {INPUT_IMAGE}")
    print(f"   Size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Step 1: Preprocess for marker detection
    print(f"\n{'STEP 1: PREPROCESSING FOR MARKERS':-^80}")
    thresh_detection = preprocess_for_detection(img)
    cv2.imwrite("test_01_thresh_detection.png", thresh_detection)
    print("   ✓ Saved: test_01_thresh_detection.png")
    
    # Step 2: Detect markers and warp
    print(f"\n{'STEP 2: DETECTING MARKERS':-^80}")
    debug_img = img.copy()
    warped = find_paper(thresh_detection, debug_image=debug_img)
    
    cv2.imwrite("test_02_marker_detection.png", debug_img)
    print("   ✓ Saved: test_02_marker_detection.png")
    
    if warped.shape != (1414, 1000):
        print(f"\n❌ ERROR: Marker detection failed!")
        print(f"   Expected size: 1000x1414, Got: {warped.shape[1]}x{warped.shape[0]}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check test_02_marker_detection.png - are all 4 corners marked in GREEN?")
        print(f"  2. Make sure your OMR sheet has BLACK SQUARES in all 4 corners")
        print(f"  3. Try adjusting lighting or rescanning")
        return False
    
    cv2.imwrite("test_03_warped_raw.png", warped)
    print("   ✓ Saved: test_03_warped_raw.png")
    print(f"   ✓ Sheet detected successfully!")
    
    # Step 3: Clean for grading
    print(f"\n{'STEP 3: CLEANING FOR BUBBLE DETECTION':-^80}")
    clean_warped = preprocess_for_grading(warped)
    cv2.imwrite("test_04_warped_clean.png", clean_warped)
    print("   ✓ Saved: test_04_warped_clean.png")
    
    # Step 4: Detect answers
    print(f"\n{'STEP 4: DETECTING BUBBLES':-^80}")
    answers = detect_answers(clean_warped, debug=True)
    # This will save:
    # - test_05_crop_col1.png
    # - test_05_crop_col2.png  
    # - test_05_grid_overlay.png
    
    # Step 5: Grade
    print(f"\n{'STEP 5: GRADING':-^80}")
    result = grade_answers(answers, ANSWER_KEY)
    
    # Display results
    print(f"\n{'RESULTS':-^80}")
    print(f"\n  📊 SCORE: {result['score']:.1f}/100")
    print(f"  ✅ Correct:  {result['correct']:2d}/{result['total']}")
    print(f"  ❌ Wrong:    {result['wrong']:2d}/{result['total']}")
    print(f"  ⚪ Empty:    {result['empty']:2d}/{result['total']}")
    
    # Show wrong/empty answers
    print(f"\n  📝 DETAILS:")
    
    if result['wrong'] + result['empty'] == 0:
        print(f"     🎉 PERFECT SCORE! All answers correct!")
    else:
        print(f"     Wrong/Empty answers:")
        for q_num, detail in sorted(result['details'].items()):
            if detail['status'] != 'CORRECT':
                student = detail['student'] if detail['student'] else '(empty)'
                correct = detail['correct']
                print(f"       Q{q_num:2d}: You={student:6s} | Correct={correct} | [{detail['status']}]")
    
    print("="*80)
    
    # Save results
    with open("test_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Full results saved to: test_result.json")
    
    # Summary of debug files
    print(f"\n{'DEBUG FILES SAVED':-^80}")
    print(f"  📁 Visual inspection files:")
    print(f"     1. test_01_thresh_detection.png  - Are markers visible as WHITE squares?")
    print(f"     2. test_02_marker_detection.png  - Are all 4 corners marked GREEN?")
    print(f"     3. test_03_warped_raw.png        - Is sheet straight and aligned?")
    print(f"     4. test_04_warped_clean.png      - Are bubbles clear without table lines?")
    print(f"     5. debug_crop_col1.png           - Left column bubbles")
    print(f"     6. debug_crop_col2.png           - Right column bubbles")
    print(f"     7. debug_grid_overlay.png        - Grid alignment check")
    print("="*80)
    
    return True


def create_sample_answer_key_template():
    """Helper function to generate answer key template"""
    print("\n# Copy this template and fill in your correct answers:")
    print("ANSWER_KEY = {")
    for i in range(1, 31):
        print(f"    {i}: 'A',  # Question {i}")
    print("}")


if __name__ == "__main__":
    print("\n")
    
    # Check if user wants to see template
    if len(sys.argv) > 1 and sys.argv[1] == "--template":
        create_sample_answer_key_template()
        sys.exit(0)
    
    # Run test
    success = test_omr()
    
    if success:
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"\nNext steps:")
        print(f"  1. Check the debug images to verify detection")
        print(f"  2. If grid misalignment, adjust ROI coordinates in detect_answers.py")
        print(f"  3. Try with more test sheets to validate accuracy")
        print(f"  4. Once satisfied, integrate into your app!")
    else:
        print(f"\n❌ TEST FAILED - Check error messages above")
        print(f"\nNeed help?")
        print(f"  - Check debug images to see what went wrong")
        print(f"  - Verify your OMR sheet has clear black corner markers")
        print(f"  - Ensure bubbles are filled with dark pen/pencil")
    
    print("\n" + "="*80 + "\n")
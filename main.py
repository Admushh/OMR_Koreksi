from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import json

from omr_core.preprocess import preprocess_for_markers, preprocess_for_answers
from omr_core.detect_sheet import find_paper, order_points
from omr_core.detect_answers import detect_answers
from omr_core.grading import grade_answers

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANSWER_KEY_PATH = "answer_key.json"

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def load_answer_key():
    if not os.path.exists(ANSWER_KEY_PATH):
        raise HTTPException(status_code=400,
            detail="Answer key not found. Please upload key first.")
    with open(ANSWER_KEY_PATH, "r") as f:
        try:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
        except json.JSONDecodeError:
            raise HTTPException(status_code=500,
                detail="Format Answer Key rusak/salah.")


async def read_image_file(file: UploadFile) -> np.ndarray:
    """Read uploaded image file and return BGR ndarray."""
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(status_code=400,
            detail=f"Format tidak didukung: {file.content_type}. Gunakan JPG/PNG/WebP.")

    await file.seek(0)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File gambar kosong.")

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400,
            detail="Format gambar tidak valid atau file rusak.")
    return image


def find_paper_with_fallback(image: np.ndarray):


    def _try_detect(src_image):
        """Attempt marker detection on a source image. Returns (warped_ready, warped_gray) or None."""
        thresh = preprocess_for_markers(src_image)
        result = find_paper(thresh)

        if result is None:
            return None

        warped_binary, M_warp = result

        # Warp the ORIGINAL grayscale image (not binary) for bubble detection
        src_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        warped_gray = cv2.warpPerspective(src_gray, M_warp, (1000, 1414))

        # Enhance for answer detection
        warped_ready = preprocess_for_answers(warped_gray)

        return warped_ready, warped_gray

    # --- Attempt 1: direct ---
    result = _try_detect(image)
    if result is not None:
        print("[pipeline] Markers found on 1st attempt.")
        return result

    # --- Attempt 2: add white padding (handles aggressive auto-crop scanners) ---
    PAD = 50
    padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD,
                                cv2.BORDER_CONSTANT, value=[255, 255, 255])
    result = _try_detect(padded)
    if result is not None:
        print("[pipeline] Markers found after padding (auto-crop fallback).")
        return result

    return None


def process_ljk(image: np.ndarray, num_questions: int = 30, debug: bool = False):

    result = find_paper_with_fallback(image)

    if result is None:
        return None, None, None, None

    warped_ready, warped_gray = result

    # Detect answers on enhanced grayscale
    answers = detect_answers(warped_ready, num_questions=num_questions, debug=debug)

    # Perform Name & ID OCR
    from omr_core.ocr import extract_name_and_id
    student_name, student_id = extract_name_and_id(warped_gray)

    return answers, warped_ready, student_name, student_id



# ENDPOINTS

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0"}


@app.post("/upload-key")
async def upload_key(file: UploadFile = File(...)):
    image = await read_image_file(file)
    try:
        key, _, _, _ = process_ljk(image, num_questions=30, debug=False)
        if key is None:
            raise HTTPException(status_code=400,
                detail="Kertas LJK tidak terdeteksi. Pastikan foto jelas & background kontras.")

        # Save key
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump(key, f, indent=4)

        return {"message": "Key saved successfully", "key": key}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[upload-key] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/scan")
async def scan(
    file: UploadFile = File(...),
    answer_key_json: str = Form(None),
    num_questions: int = Form(30),
):
    # 1. Resolve answer key
    if answer_key_json:
        try:
            raw_key = json.loads(answer_key_json)
            answer_key = {int(k): v for k, v in raw_key.items()}
        except (json.JSONDecodeError, ValueError, KeyError):
            raise HTTPException(status_code=400,
                detail="Format kunci jawaban (JSON) tidak valid.")
    else:
        answer_key = load_answer_key()

    # 2. Read image
    image = await read_image_file(file)

    try:
        # 3. Full OMR pipeline
        student_answers, _, student_name, student_id = process_ljk(image, num_questions=num_questions, debug=False)

        if student_answers is None:
            raise HTTPException(status_code=400,
                detail="Kertas LJK tidak terdeteksi. Pastikan foto jelas & 4 marker sudut terlihat.")

        # 4. Grade
        result = grade_answers(student_answers, answer_key)

        # 5. Build response 
        details_list = []
        if "details" in result:
            for q_num, info in result["details"].items():
                details_list.append({
                    "question_no": int(q_num),
                    "student_answer": str(info["student"]) if info["student"] else "-",
                    "correct_answer": str(info["correct"]) if info["correct"] else "?",
                    "status": str(info["status"]),
                })
        details_list.sort(key=lambda x: x["question_no"])

        return {
            "score": result.get("score", 0),
            "student_name": student_name,
            "student_id": student_id,
            "summary": result.get("summary", {}),
            "student_answers": student_answers,
            "details": details_list,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[scan] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses LJK: {str(e)}")

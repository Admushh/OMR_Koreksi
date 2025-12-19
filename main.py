from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import os

from omr.preprocess import preprocess_image
from omr.detect_sheet import find_paper
from omr.detect_answers import detect_answers
from omr.grading import grade_answers

app = FastAPI()

ANSWER_KEY_PATH = "answer_key.txt"


def load_answer_key():
    if not os.path.exists(ANSWER_KEY_PATH):
        raise HTTPException(
            status_code=400,
            detail="Answer key not found. Upload answer key first."
        )
    with open(ANSWER_KEY_PATH, "r") as f:
        return f.read().split(",")


@app.post("/upload-key")
async def upload_answer_key(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    processed = preprocess_image(image)
    warped = find_paper(processed)
    key_answers = detect_answers(warped)

    with open(ANSWER_KEY_PATH, "w") as f:
        f.write(",".join(key_answers))

    return {
        "message": "Answer key saved",
        "total_questions": len(key_answers),
        "key": key_answers
    }


@app.post("/scan")
async def scan_omr(file: UploadFile = File(...)):
    answer_key = load_answer_key()

    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    processed = preprocess_image(image)
    warped = find_paper(processed)
    answers = detect_answers(warped)

    result = grade_answers(answers, answer_key)

    return {
        "answers": answers,
        "result": result
    }

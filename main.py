from fastapi import FastAPI, UploadFile, File, HTTPException
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
        raise HTTPException(status_code=400, detail="Answer key not found")
    with open(ANSWER_KEY_PATH) as f:
        return f.read().split(",")


@app.post("/upload-key")
async def upload_key(file: UploadFile = File(...)):
    image = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    processed = preprocess_image(image)
    cv2.imwrite("debug_preprocess.png", processed)

    warped = find_paper(processed)
    cv2.imwrite("debug_warped.png", warped)

    key = detect_answers(warped)

    with open(ANSWER_KEY_PATH, "w") as f:
        f.write(",".join(key))

    return {"message": "Key saved", "key": key}


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    answer_key = load_answer_key()

    image = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    processed = preprocess_image(image)
    warped = find_paper(processed)

    answers = detect_answers(warped)
    result = grade_answers(answers, answer_key)

    return {
        "answers": answers,
        "result": result
    }

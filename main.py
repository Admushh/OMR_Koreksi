from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from omr.preprocess import preprocess_image
from omr.detect_sheet import find_paper
from omr.detect_answers import detect_answers
from omr.grading import grade_answers

app = FastAPI()

ANSWER_KEY = [
    "A","C","B","D","A",
    "B","C","D","A","E"
]

@app.post("/scan")
async def scan_omr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    processed = preprocess_image(image)
    warped = find_paper(processed)
    answers = detect_answers(warped)
    result = grade_answers(answers, ANSWER_KEY)

    return result

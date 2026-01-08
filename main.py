from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import json # <--- Wajib import ini

# Import modul OMR kamu
from omr_core.preprocess import preprocess_image
from omr_core.detect_sheet import find_paper
from omr_core.detect_answers import detect_answers
# Pastikan grade_answers ada di grading.py atau detect_answers.py (sesuaikan importnya)
from omr_core.grading import grade_answers 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Boleh diakses dari mana aja (termasuk https://flutterflow.io)
    allow_credentials=True,
    allow_methods=["*"],  # Boleh semua method (POST, GET, OPTIONS, dll)
    allow_headers=["*"],  # Boleh semua header
)

# Kita ganti ekstensi jadi .json biar jelas
ANSWER_KEY_PATH = "answer_key.json" 

def load_answer_key():
    """Load kunci jawaban dari file JSON dan ubah key string jadi integer."""
    if not os.path.exists(ANSWER_KEY_PATH):
        raise HTTPException(status_code=400, detail="Answer key not found. Please upload key first.")
    
    with open(ANSWER_KEY_PATH, "r") as f:
        try:
            data = json.load(f)
            # JSON menyimpan key sebagai string ("1": "A"), kita perlu convert jadi int (1: "A")
            return {int(k): v for k, v in data.items()}
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Format Answer Key rusak/salah. Upload ulang key.")


@app.post("/upload-key")
async def upload_key(file: UploadFile = File(...)):
    # 1. Baca Gambar
    image = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(status_code=400, detail="File gambar tidak valid")

    # 2. Proses OMR
    processed = preprocess_image(image)
    warped = find_paper(processed)
    
    # 3. Deteksi Jawaban (Hasilnya Dictionary {1:'A', 2:'B'...})
    key = detect_answers(warped, num_questions=30, debug=False)

    # 4. Simpan sebagai JSON
    with open(ANSWER_KEY_PATH, "w") as f:
        json.dump(key, f, indent=4)

    return {"message": "Key saved successfully", "key": key}


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    # 1. Load Kunci Jawaban (Format Dictionary)
    answer_key = load_answer_key()

    # 2. Baca Gambar Siswa
    image = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(status_code=400, detail="File gambar tidak valid")

    # 3. Proses OMR
    processed = preprocess_image(image)
    warped = find_paper(processed)

    # 4. Deteksi & Nilai
    student_answers = detect_answers(warped, num_questions=30, debug=False)
    
    # Fungsi grade_answers sekarang menerima Dictionary vs Dictionary (Aman!)
    result = grade_answers(student_answers, answer_key)

    return {
        "score": result['score'],
        "summary": {
            "correct": result['correct'],
            "wrong": result['wrong'],
            "empty": result['empty']
        },
        "student_answers": student_answers,
        "details": result['details']
    }
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import json

# ==========================================
# IMPORT MODUL OMR (Sesuaikan dengan struktur folder lu)
# ==========================================
from omr_core.preprocess import preprocess_image
from omr_core.detect_sheet import find_paper
from omr_core.detect_answers import detect_answers
from omr_core.grading import grade_answers 

app = FastAPI()

# ==========================================
# SETUP CORS (Biar bisa diakses FlutterFlow)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANSWER_KEY_PATH = "answer_key.json" 

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_answer_key():
    """Load kunci jawaban dari file JSON lokal di server."""
    if not os.path.exists(ANSWER_KEY_PATH):
        raise HTTPException(status_code=400, detail="Answer key not found. Please upload key first.")
    
    with open(ANSWER_KEY_PATH, "r") as f:
        try:
            data = json.load(f)
            # JSON key selalu string ("1"), convert jadi int (1)
            return {int(k): v for k, v in data.items()}
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Format Answer Key rusak/salah.")

async def read_image_file(file: UploadFile):
    """Fungsi aman untuk membaca file gambar dari upload."""
    try:
        # 1. Pastikan cursor file ada di awal
        await file.seek(0)
        
        # 2. Baca isi file
        contents = await file.read()
        
        # 3. Cek apakah kosong
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File gambar kosong.")
            
        # 4. Convert ke numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # 5. Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Format gambar tidak valid/rusak.")
            
        return image
    except Exception as e:
        print(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail="Gagal membaca file gambar.")

# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/upload-key")
async def upload_key(file: UploadFile = File(...)):
    # Baca gambar dengan aman
    image = await read_image_file(file)

    # Proses OMR untuk Kunci Jawaban
    try:
        processed = preprocess_image(image)
        warped = find_paper(processed)
        
        if warped is None:
             raise HTTPException(status_code=400, detail="Kertas LJK tidak terdeteksi. Pastikan foto jelas & background kontras.")

        # Deteksi jawaban (Misal: {1: 'A', 2: 'B'})
        key = detect_answers(warped, num_questions=30, debug=False)

        # Simpan ke file JSON
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump(key, f, indent=4)

        return {"message": "Key saved successfully", "key": key}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/scan")
async def scan(
    file: UploadFile = File(...), 
    answer_key_json: str = Form(None) # Parameter opsional dari Flutter
):
    # 1. Tentukan Kunci Jawaban (Dari Parameter atau File)
    if answer_key_json:
        try:
            raw_key = json.loads(answer_key_json)
            # Pastikan key jadi integer
            answer_key = {int(k): v for k, v in raw_key.items()}
        except:
             raise HTTPException(status_code=400, detail="Format kunci jawaban (JSON) tidak valid")
    else:
        answer_key = load_answer_key()

    # 2. Baca Gambar Siswa dengan Aman
    image = await read_image_file(file)

    try:
        # 3. Proses OMR
        processed = preprocess_image(image)
        warped = find_paper(processed)
        
        if warped is None:
             raise HTTPException(status_code=400, detail="Kertas LJK tidak terdeteksi.")

        student_answers = detect_answers(warped, num_questions=30, debug=False)
        
        # 4. Grading / Penilaian
        result = grade_answers(student_answers, answer_key)

        # 5. Transformasi Data untuk FlutterFlow
        # Mengubah Dictionary 'details' menjadi List of Dictionary
        details_list = []
        
        # Loop hasil grading (format: {1: {'student': 'A', 'status': 'WRONG'}, ...})
        if 'details' in result:
            for q_num, info in result['details'].items():
                details_list.append({
                    "question_no": int(q_num),
                    "student_answer": str(info['student']) if info['student'] else "-", # Handle None
                    "correct_answer": str(info['correct']) if info['correct'] else "?", # Handle None
                    "status": str(info['status'])
                })

        # Urutkan berdasarkan nomor soal
        details_list.sort(key=lambda x: x['question_no'])

        # 6. Return Response
        return {
            "score": result.get('score', 0),
            "summary": result.get('summary', {}),
            "student_answers": student_answers, # Tetap kirim raw data buat debug
            "details": details_list # <--- INI YANG DIPAKE DI FLUTTERFLOW (LIST)
        }

    except Exception as e:
        print(f"Error processing scan: {e}") # Print error di terminal biar kebaca
        raise HTTPException(status_code=500, detail=f"Gagal memproses LJK: {str(e)}")


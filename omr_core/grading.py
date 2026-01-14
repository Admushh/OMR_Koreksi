def grade_answers(student_answers, answer_key):
    """
    Grades the detected answers against the answer key.

    Args:
        student_answers (dict): Dictionary of student's answers {1: 'A', 2: 'B', ...}
        answer_key (dict): Dictionary of correct answers {1: 'A', 2: 'C', ...}

    Returns:
        dict: A dictionary containing the score, summary (counts), and detailed results.
    """
    correct = 0
    wrong = 0
    empty = 0
    details = {}
    
    # Use the answer key's questions as the source of truth for total questions
    total_questions = len(answer_key)
    
    for q_num in sorted(answer_key.keys()):
        student_ans = student_answers.get(q_num) # Use .get() for safety
        correct_ans = answer_key.get(q_num)
        
        status = ""
        # Cek jika None atau string '-'
        if student_ans is None or student_ans == '-':
            empty += 1
            status = "EMPTY"
        elif student_ans == correct_ans:
            correct += 1
            status = "CORRECT"
        else:
            wrong += 1
            status = "WRONG"
        
        details[q_num] = {'student': student_ans, 'correct': correct_ans, 'status': status}
    
    score = (correct / total_questions) * 100 if total_questions > 0 else 0

    # --- PERUBAHAN UTAMA DI SINI ---
    # Kita bungkus correct, wrong, empty ke dalam dictionary 'summary'
    # biar main.py bisa bacanya pas dipanggil result.get('summary')
    summary_data = {
        'correct': correct, 
        'wrong': wrong, 
        'empty': empty, 
        'total': total_questions
    }

    return {
        'score': round(score, 2), 
        'summary': summary_data, # <--- INI KUNCINYA BANG!
        'details': details
    }
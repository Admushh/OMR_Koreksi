def grade_answers(student_answers, answer_key):
    total = min(len(student_answers), len(answer_key))
    correct = 0
    wrong = 0
    empty = 0

    detail = []

    for i in range(total):
        s = student_answers[i]
        k = answer_key[i]

        if s == "-":
            empty += 1
            is_correct = False
        elif s == k:
            correct += 1
            is_correct = True
        else:
            wrong += 1
            is_correct = False

        detail.append({
            "question": i + 1,
            "student": s,
            "key": k,
            "correct": is_correct
        })

    score = (correct / total) * 100

    return {
        "score": round(score, 2),
        "correct": correct,
        "wrong": wrong,
        "empty": empty,
        "total": total,
        "detail": detail
    }

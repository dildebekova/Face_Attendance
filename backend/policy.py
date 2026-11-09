import math

def decide(scores, threshold, margin):
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    if not ranked or not math.isfinite(ranked[0][1]) or ranked[0][1] < threshold:
        raise ValueError("Лицо не распознано. Повторите проверку или обратитесь к преподавателю.")
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
        raise ValueError("Неоднозначное совпадение. Нужна ручная проверка.")
    return ranked[0][0]

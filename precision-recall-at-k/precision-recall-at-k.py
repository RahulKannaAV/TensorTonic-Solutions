def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    intersect_val = 0
    rel_set = set(relevant)
    for i in range(k):
        if(recommended[i] in rel_set):
            intersect_val += 1

    prec_val = intersect_val/k
    rec_val = intersect_val/len(relevant)

    return [prec_val, rec_val]
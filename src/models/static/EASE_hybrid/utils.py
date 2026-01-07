def build_valid_lists(valid_gt, rec):
    actual, pred = [], []
    for u in range(len(rec)):
        if len(valid_gt[u]) == 0:
            continue
        actual.append(valid_gt[u])
        pred.append(rec[u].tolist())
    return actual, pred

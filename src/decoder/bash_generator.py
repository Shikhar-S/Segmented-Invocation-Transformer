from utils.metric_utils import compute_metric
import numpy as np

def get_score(truth_list,pred_list):
    scores = []
    all_scores = []
    c=0
    batch_size = len(truth_list)
    assert ( len(truth_list) == len(pred_list) )
    for b in range(batch_size):
        truth = truth_list[b][0]
        preds = pred_list[b]
        max_score = -1
        inner_scores = []
        min_above_zero = False
        for p in preds  :
            score = compute_metric(p,1,truth,{'u1':1.0,'u2':1.0})
            inner_scores.append(score)
            if score>0:
                min_above_zero = True
            max_score = max(score,max_score)
        if min_above_zero:
            scores.append(max_score)
        else:
            av = sum(inner_scores)/len(inner_scores)
            scores.append(av)

        all_scores.append(inner_scores)
    scores = np.array(scores)
    all_scores = np.array(all_scores)
    return scores,all_scores
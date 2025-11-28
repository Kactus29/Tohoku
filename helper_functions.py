import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score




NUM_CLASSES = 3
VESSEL_CLASSES = [1, 2]





# =========================
#          INIT
# =========================

def init_confmat_sums(num_classes=NUM_CLASSES) :
    sums = {
        "TP" : np.zeros(num_classes, dtype=np.int64),
        "FP" : np.zeros(num_classes, dtype=np.int64),
        "FN" : np.zeros(num_classes, dtype=np.int64),
        "TN" : np.zeros(num_classes, dtype=np.int64),
        "N"  : 0
    }
    return sums

def init_auc_buffers(include_classes=VESSEL_CLASSES) :
    return {c : {"y_true" : [], "y_score" : []} for c in include_classes}



# =========================
#         UPDATE
# =========================

def update_confmat_sums(sums, preds, targets, num_classes=NUM_CLASSES) :
    with torch.no_grad() :
        for c in range(num_classes) :
            p = (preds == c)
            t = (targets == c)
            tp = (p & t).sum().item()
            fp = (p & ~t).sum().item()
            fn = (~p & t).sum().item()
            tn = (~p & ~t).sum().item()
            sums["TP"][c] += tp
            sums["FP"][c] += fp
            sums["FN"][c] += fn
            sums["TN"][c] += tn
        sums["N"] += targets.numel()



def update_auc_buffers(buffers, logits, targets):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        flat_probs = probs.permute(0,2,3,1).reshape(-1, probs.shape[1])
        flat_t     = targets.reshape(-1)

        for c in buffers.keys():
            y_true = (flat_t == c).cpu().numpy().astype(np.uint8)
            y_score = flat_probs[:, c].cpu().numpy()
            buffers[c]["y_true"].append(y_true)
            buffers[c]["y_score"].append(y_score)




# =========================
#         COMPUTE
# =========================

def compute_scalar_metrics_from_confmat(sums, include_classes):
    TP, FP, FN, TN = sums["TP"], sums["FP"], sums["FN"], sums["TN"]
    eps = 1e-7

    acc_c  = (TP + TN) / np.maximum(TP + TN + FP + FN, eps)
    prec_c = TP / np.maximum(TP + FP, eps)
    rec_c  = TP / np.maximum(TP + FN, eps)
    spec_c = TN / np.maximum(TN + FP, eps)
    dice_c = (2 * TP) / np.maximum(2 * TP + FP + FN, eps)
    iou_c  = TP / np.maximum(TP + FP + FN, eps)

    idx = np.array(include_classes, dtype=int)
    return {
        "accuracy":   float(np.mean(acc_c[idx])),
        "precision":  float(np.mean(prec_c[idx])),
        "recall":     float(np.mean(rec_c[idx])),
        "specificity":float(np.mean(spec_c[idx])),
        "dice":       float(np.mean(dice_c[idx])),
        "iou":        float(np.mean(iou_c[idx])),
    }




def compute_mean_auc(buffers):
    aucs = []
    for c, pack in buffers.items():
        if len(pack["y_true"]) == 0: 
            continue
        y_true = np.concatenate(pack["y_true"])
        y_score = np.concatenate(pack["y_score"])
        pos = (y_true == 1).sum()
        neg = (y_true == 0).sum()
        if pos == 0 or neg == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_true, y_score))
        except:
            pass
    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))
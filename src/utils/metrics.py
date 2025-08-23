import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def full_report(y_true, y_pred, target_names):
    rep = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return rep, cm
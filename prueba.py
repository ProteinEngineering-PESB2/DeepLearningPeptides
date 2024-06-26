import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

v = np.ones(10)
vp = np.zeros(10)

vp = np.concatenate((v,vp), axis=0)
y_pred = random.choices(vp, k=20)
print(y_pred)

precision = precision_score(vp, y_pred, zero_division=0.0, average='binary')
recall = recall_score(vp, y_pred, zero_division=0.0, average='binary')
f1 = f1_score(vp, y_pred, zero_division=0.0,average='binary')
tn, fp, fn, tp = confusion_matrix(vp, y_pred).ravel()
mcc = matthews_corrcoef(vp, y_pred)


print(precision, recall, f1,mcc)
print(tn, fp, fn, tp)

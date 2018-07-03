import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    print("\n Confusion Matrix:\n")
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "ant"]
y_pred = ["cat", "ant", "cat", "cat", "ant", "cat", "bird", "bird"]
conf = confusion_matrix(y_true, y_pred, labels=["bird", "cat", "ant"])
clas = classification_report(y_true, y_pred,target_names=["ant", "bird", "cat"])
cohe = cohen_kappa_score(y_true, y_pred, labels=["ant", "bird", "cat"])

print_cm(conf, labels=["bird", "cat", "ant"])
print ('\n Classification Report: \n' ,clas, '\n Cohen Kappa Score:',cohe, '\n')


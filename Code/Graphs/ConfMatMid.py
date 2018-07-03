import csv
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    out.write("\n Confusion Matrix:\n")
    # Print header
    out.write("      " + empty_cell)
    for label in labels:
        out.write("%{0}s".format(columnwidth) % label+ " ")
    out.write("\n")
    # Print rows
    for i, label1 in enumerate(labels):
        out.write("    %{0}s".format(columnwidth) % label1+ " ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            out.write(cell+ " ")
        out.write("\n")

location = 'Middle_Data_Cop.csv'
df = pd.read_csv(location)
labels_det = ["No", "Damage", "unknown"]
out = open("output/Conf_mat.txt", "a") 

condi = [
    (df["_damage"] == "partial" ), (df["_damage"] == "none"),
    (df["_damage"] == "significant"), (df["_damage"] == "destroyed")]
choic = ['No', 'No', 'Damage', 'Damage']
df["Dam_det"] = np.select(condi, choic, default="unknown")

keyList = df.keys().tolist()
detThresList = np.around(np.linspace(0.0, 1.0, 101), decimals=2)
for key in keyList:
    if key[-4:] == "mean" or key[-6:] == "median":
        for thres in detThresList:
            df[key + "_" + str(thres)] = np.where(df[key] > thres, 'Damage', 'No')

            conf = confusion_matrix(df["Dam_det"], df[key + "_" + str(thres)], labels=labels_det)
            clas = classification_report(df["Dam_det"], df[key + "_" + str(thres)],target_names=labels_det)
            cohe = cohen_kappa_score(df["Dam_det"], df[key + "_" + str(thres)], labels=labels_det)

            out.write("\n " + key + " " + str(thres) + ": \n")
            print_cm(conf, labels=labels_det)
            out.write('\n Classification Report: \n' + str(clas) + '\n Cohen Kappa Score:' + str(cohe) + '\n')
            with open("output/CohenScore.csv", "a", newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([key + " " + str(thres), cohe])


condiCop = [
    (df["join_grading"] == "Completely Destroyed" ), (df["join_grading"] == "Highly Damaged"),
    (df["join_grading"] == "Moderately Damaged"), (df["join_grading"] == "Negligible to slight damage"),
    (df["join_grading"] == "Not Affected")]
choicCop = ['Damage', 'Damage', 'Damage', 'No', 'No']
df["Dam_Cop"] = np.select(condiCop, choicCop, default="unknown")

conf = confusion_matrix(df["Dam_det"], df["Dam_Cop"], labels=labels_det)
clas = classification_report(df["Dam_det"], df["Dam_Cop"],target_names=labels_det)
cohe = cohen_kappa_score(df["Dam_det"], df["Dam_Cop"], labels=labels_det)

out.write("\n Copernicus: \n")
print_cm(conf, labels=labels_det)
out.write('\n Classification Report: \n' + str(clas) + '\n Cohen Kappa Score:' + str(cohe) + '\n')
with open("output/CohenScore.csv", "a", newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(["Copernicus", cohe])

writer = pd.ExcelWriter("output/dataframe.xlsx")
df.to_excel(writer, 'data')
writer.save()
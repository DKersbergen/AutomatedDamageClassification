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
labels_class = ["none", "partial", "significant", "destroyed", "unknown"]
out = open("output/Conf_mat_class3.txt", "a") 

keyList = df.keys().tolist()
detThresList = np.around(np.linspace(0.0, 0.94, 38), decimals=2)
counter = 0
for key in keyList:
    if key[-6:] == "median":
        for thres in detThresList:
            detThres1 = np.around(np.linspace(thres, 0.96, int((1.0-thres)*39)), decimals=2)
            for thres1 in detThres1:                
                detThres2 = np.around(np.linspace(thres1, 0.98, int((1.0-thres1)*40)), decimals=2)
                for thres2 in detThres2:
                    print(thres, thres1, thres2)
                    counter += 1
                    condis = [
                        (df[key] <= thres ), (df[key] > thres) & (df[key] <= thres1),
                        (df[key] > thres1) & (df[key] <= thres2), (df[key] > thres2)]
                    choics = ['none', 'partial', 'significant', 'destroyed']
                    dfstring = key + "_" + str(thres) +"_" + str(thres1) +"_" + str(thres2)
                    df[dfstring] = np.select(condis, choics, default="unknown")
                    conf = confusion_matrix(df["_damage"], df[dfstring], labels=labels_class)
                    clas = classification_report(df["_damage"], df[dfstring],target_names=labels_class)
                    cohe = cohen_kappa_score(df["_damage"], df[dfstring], labels=labels_class)

                    out.write("\n " + dfstring + ": \n")
                    print_cm(conf, labels_class)
                    out.write('\n Classification Report: \n' + str(clas) + '\n Cohen Kappa Score:' + str(cohe) + '\n')
                    counter+=1
                    print(counter)
                    with open("output/CohenScore_class3.csv", "a", newline='') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',')
                        csvwriter.writerow([dfstring, cohe])
                        
condiCop = [
    (df["join_grading"] == "Completely Destroyed" ), (df["join_grading"] == "Highly Damaged"),
    (df["join_grading"] == "Moderately Damaged"), (df["join_grading"] == "Negligible to slight damage"),
    (df["join_grading"] == "Not Affected")]
choicCop = ['destroyed', 'significant', 'partial', 'none', 'none']
df["Dam_Cop"] = np.select(condiCop, choicCop, default="unknown")

conf = confusion_matrix(df["_damage"], df["Dam_Cop"], labels=labels_class)
clas = classification_report(df["_damage"], df["Dam_Cop"],target_names=labels_class)
cohe = cohen_kappa_score(df["_damage"], df["Dam_Cop"], labels=labels_class)

out.write("\n Copernicus: \n")
print_cm(conf, labels=labels_class)
out.write('\n Classification Report: \n' + str(clas) + '\n Cohen Kappa Score:' + str(cohe) + '\n')
with open("output/CohenScore_class3.csv", "a", newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(["Copernicus", cohe])

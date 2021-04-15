import spacy
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import csv
from pandas import DataFrame
from statistics import mean

dataset = pd.read_csv("SemEval2018-Task3/datasets/train/SemEval2018-T3-train-taskB.txt",sep='\t',header=0,index_col=0,quoting=csv.QUOTE_NONE, error_bad_lines=False)
documents = dataset['Tweet text']

nlp = spacy.load('en_core_web_sm')

'''
1.	Class Distributions 
'''
dataset_frequency = dataset.groupby(dataset['Label']).size()
print('The number of instances for each of the four classification labels in the training set:')
print(dataset_frequency)
print()
dataset_frequency_percentage = dataset.groupby(dataset['Label']).size()/len(dataset)*100.0
print('Training Set Relative Label Frequency (%)')
print(dataset_frequency_percentage)
print()

'''
2.	Baselines 
'''
test_set = pd.read_csv("SemEval2018-Task3/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt",sep='\t',header=0,index_col=0,quoting=csv.QUOTE_NONE, error_bad_lines=False)
test_set_true_label = test_set['Label'].tolist()

precisiondf = DataFrame()
f1df= DataFrame()
recalldf = DataFrame()
accuracydf= DataFrame()

for i in range(100):
    np.random.seed(i)
    random_label = np.random.randint(0,4,size=len(test_set_true_label))
    precision = precision_score(test_set_true_label, random_label,labels=[0,1,2,3], average=None)
    recall = recall_score(test_set_true_label, random_label, labels=[0,1,2,3],average=None)
    f1 = f1_score(test_set_true_label, random_label, labels=[0,1,2,3],average=None)
    random_confusion_matrix = confusion_matrix(test_set_true_label, random_label, labels=[0, 1, 2, 3])
    random_confusion_matrix = random_confusion_matrix.astype('float') / random_confusion_matrix.sum(axis=1)[:, np.newaxis]
    accuracy = random_confusion_matrix.diagonal()
    precisiondf = precisiondf.append(pd.Series(precision), ignore_index=True)
    recalldf = recalldf.append(pd.Series(recall), ignore_index=True)
    f1df = f1df.append(pd.Series(f1), ignore_index=True)
    accuracydf = accuracydf.append(pd.Series(accuracy), ignore_index=True)

print("Accuracy Score for Class 0: {}, Class 1: {}, Class 2: {}, Class 3: {}".format( mean(list(accuracydf[0])), mean(list(accuracydf[1])), mean(list(accuracydf[2])), mean(list(accuracydf[3])) ))
print("Precision Score for Class 0: {}, Class 1: {}, Class 2: {}, Class 3: {}".format( mean(list(precisiondf[0])), mean(list(precisiondf[1])), mean(list(precisiondf[2])), mean(list(precisiondf[3])) ))
print("Recall Score for Class 0: {}, Class 1: {}, Class 2: {}, Class 3: {}".format( mean(list(recalldf[0])), mean(list(recalldf[1])), mean(list(recalldf[2])), mean(list(recalldf[3])) ))
print("F1 Score for Class 0: {}, Class 1: {}, Class 2: {}, Class 3: {}".format( mean(list(f1df[0])), mean(list(f1df[1])), mean(list(f1df[2])), mean(list(f1df[3])) ))
print()
print("Accuracy macro average: {}".format((mean(list(accuracydf[0]))+ mean(list(accuracydf[1]))+ mean(list(accuracydf[2]))+mean(list(accuracydf[3])))/4))
print("Precision macro average: {}".format((mean(list(precisiondf[0]))+ mean(list(precisiondf[1]))+ mean(list(precisiondf[2]))+mean(list(precisiondf[3])))/4))
print("Recall macro average: {}".format((mean(list(recalldf[0]))+ mean(list(recalldf[1]))+ mean(list(recalldf[2]))+mean(list(recalldf[3])))/4))
print("F1 macro average: {}".format((mean(list(f1df[0]))+ mean(list(f1df[1]))+ mean(list(f1df[2]))+mean(list(f1df[3])))/4))
test_frequency_percentage = test_set.groupby(test_set['Label']).size()/len(test_set)*100
print('Test Set Relative Label Frequency (%)')
print(test_frequency_percentage)
print()
print("Accuracy weighted average: {}".format((0.60331633*mean(list(accuracydf[0]))+ 0.20918367*mean(list(accuracydf[1]))+ 0.10841837*mean(list(accuracydf[2]))+ 0.07908163*mean(list(accuracydf[3])))))
print("Precision weighted average: {}".format((0.60331633*mean(list(precisiondf[0]))+ 0.20918367*mean(list(precisiondf[1]))+ 0.10841837*mean(list(precisiondf[2]))+0.07908163*mean(list(precisiondf[3])))))
print("Recall weighted average: {}".format((0.60331633*mean(list(recalldf[0]))+ 0.20918367*mean(list(recalldf[1]))+ 0.10841837*mean(list(recalldf[2]))+0.07908163*mean(list(recalldf[3])))))
print("F1 weighted average: {}".format((0.60331633*mean(list(f1df[0]))+ 0.20918367*mean(list(f1df[1]))+ 0.10841837*mean(list(f1df[2]))+0.07908163*mean(list(f1df[3])))))
print()

majority_label = np.zeros(len(test_set_true_label))
majority_report = classification_report(test_set_true_label, majority_label, target_names=['0', '1', '2','3'])
print("Classification Report for Majority Baseline:")
print(majority_report)

majority_confusion_matrix = confusion_matrix(test_set_true_label, majority_label, labels=[0, 1, 2, 3])
majority_confusion_matrix = majority_confusion_matrix.astype('float') / majority_confusion_matrix.sum(axis=1)[:, np.newaxis]
# The diagonal entries are the accuracies of each class
print('Confusion Matrix for Majority Baseline:')
print(majority_confusion_matrix)
print()
majority_label_macro_average_accuracy = sum(majority_confusion_matrix.diagonal()) / len(majority_confusion_matrix.diagonal())
print("Macro Average Accuracy for Majority Baseline: " + str(majority_label_macro_average_accuracy) + '\n')
majority_label_weighted_averag_accuracy =  (0.60331633*1.00 + 0.20918367*0.0 + 0.10841837*0.0 + 0.07908163*0.0)
print("Weighted Average Accuracy for Majority Baseline: " + str(majority_label_weighted_averag_accuracy) + '\n')

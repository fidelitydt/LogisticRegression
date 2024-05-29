# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:50:37 2021

@author: 160010321
"""
####Code for Logistic Regression using RDM Example

from random import seed
from random import randrange
from csv import reader
from math import exp
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import softmax


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            #row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

##Add softmax function here
def softmax_fn(X):
    result = softmax(X)
    # report the probabilities
    print('softmax')
    print(result)
    # report the sum of the probabilities
    #print(sum(result))
    
    

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    epoch_list.clear()
    error_list.clear()
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            sum_error += error**2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
                #coeff=coeff*l_rate*error*x
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        epoch_list.append(epoch)
        error_list.append(sum_error)
    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    expected=list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    i=0
    for row in test:
        yhat = predict(row, coef)
        yhat_round = round(yhat)
        predictions.append(yhat_round)
        expected.append(round(row[-1]))
        #print("Time Step:",i)
        #print(i,"Expected=%d, Predicted=%.3f [%d]" % (round(row[-1]), yhat, round(yhat)))
        i=i+1
    confusion_matrix_calculation(expected,predictions)
    calculate_Accuracy_Precision_Recall(expected,predictions)
    return(predictions)
    
##Confusion Matrix
def confusion_matrix_calculation(expected,predictions):
    cnf_matrix = metrics.confusion_matrix(expected, predictions)
    print(cnf_matrix)
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
###Calculating accuracy, precision and recall
def calculate_Accuracy_Precision_Recall(expected, predicted):
    print("Accuracy:",metrics.accuracy_score(expected, predicted))
    print("Precision:",metrics.precision_score(expected, predicted))
    print("Recall:",metrics.recall_score(expected, predicted))
    print("F1 score:",metrics.f1_score(expected,predicted))

##Plot ROC Curve    
def ROC_Curve(predicted,predicted_prob):
    fpr, tpr, _ = metrics.roc_curve(predicted,  predicted_prob)
    auc = metrics.roc_auc_score(predicted, predicted_prob)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

def learning_rate_plot(dataset):
    score=list()
    width1 = 12
    height1 = 10
    width_height_1 = (width1, height1)
    plt.figure(figsize=width_height_1)
    learning_rates = [0.001, 0.003, 0.005,0.007,0.009, 0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5,0.7,0.9]
    #learning_rates = [0.1,0.3,0.5,0.7,0.9]
  
    for i in learning_rates:
        print ("learning rate is: ",i)
        scores = evaluate_algorithm(dataset, logistic_regression, n_folds,i, n_epoch)
        print(scores)
        score.append(scores)
        plt.plot(error_list, label="learning_rate:"+str(i))
        plt.ylabel('Error')
        plt.xlabel('Iterations')
    
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    print(score)
# Test the logistic regression algorithm on the RDM dataset
seed(1)
# load and prepare training data
filename ='./RegressionMPScenario1_combined.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# load and prepare test data
filename_test ='./RegressionMPScenario15.csv'
dataset_test = load_csv(filename_test)
for i in range(len(dataset_test[0])):
	str_column_to_float(dataset_test, i)
# normalize
minmax = dataset_minmax(dataset_test)
normalize_dataset(dataset_test, minmax)
# evaluate algorithm
epoch_list=list()
error_list=list()
n_folds = 2
l_rate = 0.05
#n_epoch = 6000
n_epoch = 20000
#scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

###plot for learning rate
learning_rate_plot(dataset)
#predictions_test=logistic_regression(dataset,dataset_test,l_rate,n_epoch)


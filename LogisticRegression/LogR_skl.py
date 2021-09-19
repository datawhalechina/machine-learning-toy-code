#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-08-31 18:12:44
LastEditor: JiangJi
LastEditTime: 2021-08-31 18:16:26
Discription: 
Environment: 
'''
import sys
from pathlib import Path
curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path) # add current terminal path to sys.path

from Mnist.load_data import load_local_mnist

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

(X_train, y_train), (X_test, y_test) = load_local_mnist(normalize = False,one_hot = False)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred)) # 打印报告
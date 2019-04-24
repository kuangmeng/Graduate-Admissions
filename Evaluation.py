#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:08:04 2019

@author: kuangmeng
"""
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score


def R2_score(y_test, y_head):
    score = 0.0
    # Finish it.
    score=r2_score(y_test,y_head)
    return score

def PPF(y_test, y_head):
    precise=0.0
    recall=0.0
    f1 = 0.0
    precise=precision_score(y_test, y_head)
    recall=recall_score(y_test,y_head)
    f1=f1_score(y_test,y_head)
    # Finish it.
    return precise, recall, f1

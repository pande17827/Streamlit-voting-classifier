
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


class Run:

    
    def __init__(self,X_train,X_test,y_train,y_test,voting_type,estimators) -> None:
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

        self.voting_type=voting_type
        self.estimators=estimators

        self.create_estimator()


    def create_estimator(self):
        

        algos = []
        self.algos=algos

        if 'KNN' in self.estimators:
            knn_clf = KNeighborsClassifier()
            algos.append(('knn', knn_clf))
        if 'Logistic Regression' in self.estimators:
            log_clf = LogisticRegression(solver="lbfgs", )
            algos.append(('lr', log_clf))
        if 'Gaussian Naive Bayes' in self.estimators:
            gnb_clf = GaussianNB()
            algos.append(('gnb', gnb_clf))
        if 'SVM' in self.estimators:
            if self.voting_type == "hard":
                svm_clf = SVC(gamma="scale", )
            else:
                svm_clf = SVC(gamma="scale", probability=True, )
            algos.append(('svc', svm_clf))
        if 'Random Forest' in self.estimators:
            rnd_clf = RandomForestClassifier(n_estimators=100, )
            algos.append(('rf', rnd_clf))

        
    
    def train_classifier(self):
        voting_clf = VotingClassifier(
            estimators=self.algos,
            voting=self.voting_type,)

        
            
        
        voting_clf.fit(self.X_train, self.y_train)
        y_pred = voting_clf.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        x = cross_val_score(voting_clf,self.X_train,self.y_train,cv=10,scoring='accuracy')
        cross_val=np.round(np.mean(x),2)
        

        return accuracy,cross_val


    def each_model_accuracy(self):
        model_data=[]
        for ind,model in enumerate(self.algos):
            model[1].fit(self.X_train, self.y_train)
            y_pred = model[1].predict(self.X_test)

            model_data.append(f"{self.estimators[ind]} :{accuracy_score(self.y_test, y_pred)}")
        
        print(model_data)
        print(self.estimators)

        
        return model_data




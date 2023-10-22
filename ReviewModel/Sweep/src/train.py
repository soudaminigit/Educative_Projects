import pandas as pd
import mlflow
import argparse
import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import scipy
from sklearn.ensemble import RandomForestClassifier
#from mlflow import MlflowClient

def load_data(input):
    df  = pd.read_csv(input)
    train,test = train_test_split(df,test_size=0.3)
#vec = TfidfVectorizer()
    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150)
    lbl = LabelEncoder()
#print(train['review'])
    train_vec=text_transformer.fit_transform(train['review'])
    test_vec=text_transformer.transform(test['review'])
    train_y = lbl.fit_transform(train['sentiment'])
    test_y = lbl.transform(test['sentiment'])
    return (train_vec,train_y,test_vec,test_y)


def train(params,input):
    train_X_vec,train_Y,test_X_vec,test_Y = load_data(input)
    mlflow.sklearn.autolog()
    model = RandomForestClassifier(**params)
    with mlflow.start_run() as run:
        train_model = model.fit(train_X_vec,train_Y)
        test_model = model.predict(test_X_vec)
        model_train_tfidf_score=accuracy_score(train_Y,model.predict(train_X_vec))
        print("model_train_tfidf_score :",model_train_tfidf_score)
        model_test_tfidf_score=accuracy_score(test_Y,test_model)
        print("model_train_tfidf_score :",model_test_tfidf_score)
        #Classification report for tfidf features
    report=classification_report(test_Y,test_model,target_names=['Positive','Negative'])
        
    print(report)
    


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--n_estimators", type=int, default="l50")
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--input", type=str)
    #parser.add_argument("--random_state", type=int, default=42)
    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    args = parse_args()
    params = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf
        }

    train(params,args.input)
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
#from mlflow import MlflowClient

def load_data(args):
    df  = pd.read_csv(args.input)
    train,test = train_test_split(df,test_size=0.3)
    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150)
    lbl = LabelEncoder()
    train_vec=text_transformer.fit_transform(train['review'])
    test_vec=text_transformer.transform(test['review'])
    train_y = lbl.fit_transform(train['sentiment'])
    test_y = lbl.transform(test['sentiment'])
    return (train_vec,train_y,test_vec,test_y)


def train(args):
    train_X_vec,train_Y,test_X_vec,test_Y = load_data(args)
    # Implement the training algorithm here
    


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--input", type=str)
    parser.add_argument("--random_state", type=int, default=42)
    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
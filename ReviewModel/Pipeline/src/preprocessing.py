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
import numpy
import scipy
#from mlflow import MlflowClient

def load_data(args):
    df  = pd.read_csv(args.input+"//"+"IMDB_preprocessed.csv", sep=",")
    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150)
    test_vec=text_transformer.fit_transform(df['review'])
    return (test_vec)


def train(args):
    test_vec = load_data(args)
    C=args.C,random_state=args.random_state)
    model=mlflow.pyfunc.load_model(args.model)
    output = model.predict(test_vec)
    numpy.savetxt(args.output+"//"+"Prediction_Results.csv",output)
    


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--model", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output",type=str)
    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
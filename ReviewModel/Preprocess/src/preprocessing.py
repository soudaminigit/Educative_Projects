import re
import pandas as pd
import numpy as np
import argparse
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def text_preprocessing(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = text.replace("'nt"," not")
    text = text.replace("'s"," is")
    text = re.sub("[^A-Za-z0-9]+"," ",text)
    #text = " ".join([x for x in text.split() if x not in stopwords])
    return text

def preprocess(args):
    print(args.data)
    df = pd.read_csv(args.data,sep=",")
    df = df.dropna()
    df['review'] = df['review'].map(lambda x:text_preprocessing(x))
    print("text data preprocessed !")
    df.to_csv(args.output+"/"+"IMDB_preprocessed.csv")
    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--data", type=str,help="provide IMDB csv file")
    parser.add_argument("--output", type=str,help="provide the output path")
    # parse args
    args = parser.parse_args()
    # return args
    return args


if __name__ == "__main__":
    args= parse_args()
    preprocess(args)    
    
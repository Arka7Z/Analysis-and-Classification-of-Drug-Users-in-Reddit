from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import random
from random import randint
import string
import nltk
import numpy as np
import pandas as pd
import csv

import time
import io
import collections
import math
import argparse
import pickle as pkl

 



emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=True):
    text =tokenize(s)
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    s= " ".join(stemmed_words)
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', 'url', vTEXT, flags=re.MULTILINE)
    return(vTEXT)


def remove_punctuations(text):
    translator = str.maketrans('', '', string.punctuation)
    out = text.translate(translator)
    return out



def prepro_stack(body,stemming=False):
    ret = body.lower()
    ret = remove_urls(ret)
    ret = remove_punctuations(ret)
    if (stemming):
        tokens=preprocess(ret)
        ret=" ".join(tokens)
    return ret



def getUnlabeledData(filename='first_evaluation_dataset.csv',stemming=False):
    data_rows = dict()
    data_rows['body']=[]
    data_rows['y'] = []
    with open('./../data/'+filename,'r') as csv_fp:
        csvreader = csv.reader(csv_fp, skipinitialspace=True)
        next(csvreader)
        for row in csvreader:
            tmp=prepro_stack(row[0],stemming)
            data_rows['body'].append(tmp)

    csv_fp.close()

    X_tmp=data_rows['body']


    return X_tmp

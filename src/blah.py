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
import matplotlib.pyplot as plt
import time
import io
import collections
import math
import argparse
import pickle as pkl
from tqdm import tqdm, tqdm_notebook
import statistics 
  
#######################


 
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
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens



######################################

def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', 'url', vTEXT, flags=re.MULTILINE)
    return(vTEXT)


def remove_punctuations(text):
    translator = str.maketrans('', '', string.punctuation)
    out = text.translate(translator)
    return out

def prepro_stack(body,lowercase=True):
    if lowercase:
        ret = body.lower()
    ret = remove_urls(ret)
    ret = remove_punctuations(ret)
    return ret

##############

data_rows = dict()
data_rows['body']=[]
data_rows['y'] = []
with open('./data/first_evaluation_dataset.csv','r') as csv_fp:
    csvreader = csv.reader(csv_fp, skipinitialspace=True)
    next(csvreader)
    for row in csvreader:
        tmp=prepro_stack(row[0])
        data_rows['body'].append(tmp)
        data_rows['y'].append(row[1])

with open('first_evaluation_dataset.pkl','wb') as out_fp:
    pkl.dump(data_rows,out_fp)


################################

X_tmp=data_rows['body']
Y_tmp=data_rows['y']
X=list()
Y=list()
for i in range(len(X_tmp)):
    if(Y_tmp[i]=="Addicted" or Y_tmp[i]=="Addiction-prone" or Y_tmp[i]=="Recovered" or Y_tmp[i]=="Recovering" or Y_tmp[i]=="NA" ):
        X.append(X_tmp[i])
        Y.append(Y_tmp[i])
    elif(Y_tmp[i]=="Telling about others addiction"):
        X.append(X_tmp[i])
        Y.append("NA")

data_rows['body']=X
data_rows['y']=Y



###############################

from collections import Counter
counter=Counter(Y)

counter

##############################

dataframe = pd.DataFrame.from_dict(data_rows)
dataframe.y.value_counts(normalize=True).plot(kind='bar', grid=True, figsize=(16, 9))

#######################
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(X).toarray()


# dim(features)=len(X)*num_unique_tokens
tfidf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
tfidf_dict = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf_dict), orient='index')
tfidf_dict.columns = ['tfidf']

tfidf_dict.sort_values(by=['tfidf'], ascending=False)


#############################
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=0)
svd_tfidf = svd.fit_transform(features)

svd_tfidf.shape
run = True
if run:
# run this (takes times)
    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, n_iter=500)
    tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
    print(tsne_tfidf.shape)
    tsne_tfidf_df = pd.DataFrame(tsne_tfidf)
    tsne_tfidf_df.columns = ['x', 'y']
    tsne_tfidf_df['Y'] = data_rows['y']
    tsne_tfidf_df['X'] = data_rows['body']
    tsne_tfidf_df.to_csv('./data/tsne_tfidf.csv', encoding='utf-8', index=False)
else:
# or import the dataset directly
    tsne_tfidf_df = pd.read_csv('./data/tsne_tfidf.csv')


######################################
groups = tsne_tfidf_df.groupby('Y')
fig, ax = plt.subplots(figsize=(15, 10))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', label=name)
ax.legend()
plt.show()

#####################################

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file

output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="tf-idf clustering of reddit posts",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

palette = d3['Category10'][len(tsne_tfidf_df['Y'].unique())]
color_map = bmo.CategoricalColorMapper(factors=tsne_tfidf_df['Y'].map(str).unique(), palette=palette)

plot_tfidf.scatter(x='x', y='y', color={'field': 'Y', 'transform': color_map}, 
                   legend='Y', source=tsne_tfidf_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"x": "@X", "y":"@Y"}

show(plot_tfidf)

#######################################

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
distorsions = []
sil_scores = []
k_max = 80
for k in tqdm_notebook(range(2, k_max)):
    kmeans_model = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42,  
                         init_size=1000, verbose=False, max_iter=1000)
    kmeans_model.fit(features)
    sil_score = silhouette_score(features, kmeans_model.labels_)
    sil_scores.append(sil_score)
    distorsions.append(kmeans_model.inertia_)
num_clusters = 5
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, random_state=42,                       
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000, )
kmeans = kmeans_model.fit(features)
kmeans_clusters = kmeans.predict(features)
kmeans_distances = kmeans.transform(features)



########################################

clustering=dict()
for (i, desc),category in zip(enumerate(data_rows['body']),data_rows['y']):
    clustering[kmeans_clusters[i]]=category
    if len(clustering.keys())==5:
        break
clustering

sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names()
all_keywords = []
for i in range(num_clusters):
    topic_keywords = []
    for j in sorted_centroids[i, :10]:
        topic_keywords.append(terms[j])
    all_keywords.append(topic_keywords)

keywords_df = pd.DataFrame(index=['topic_{0}'.format(i) for i in range(num_clusters)], 
                           columns=['keyword_{0}'.format(i) for i in range(10)],
                           data=all_keywords)
keywords_df

######################################

# Changing number of clusters to 3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
distorsions = []
sil_scores = []
k_max = 80
for k in tqdm_notebook(range(2, k_max)):
    kmeans_model = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42,  
                         init_size=1000, verbose=False, max_iter=1000)
    kmeans_model.fit(features)
    sil_score = silhouette_score(features, kmeans_model.labels_)
    sil_scores.append(sil_score)
    distorsions.append(kmeans_model.inertia_)
num_clusters = 4
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, random_state=42,                       
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000, )
kmeans = kmeans_model.fit(features)
kmeans_clusters = kmeans.predict(features)
kmeans_distances = kmeans.transform(features)

#######################################

clustering=dict()
for (i, desc),category in zip(enumerate(data_rows['body']),data_rows['y']):
    if kmeans_clusters[i] in clustering:
        clustering[kmeans_clusters[i]].append(category)
    else:
        clustering[kmeans_clusters[i]]=[category]
for key in clustering.keys():
    clustering[key]=statistics.mode(clustering[key])
print(clustering)
sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names()
all_keywords = []
for i in range(num_clusters):
    topic_keywords = []
    for j in sorted_centroids[i, :10]:
        topic_keywords.append(terms[j])
    all_keywords.append(topic_keywords)

keywords_df = pd.DataFrame(index=['topic_{0}'.format(i) for i in range(num_clusters)], 
                           columns=['keyword_{0}'.format(i) for i in range(10)],
                           data=all_keywords)
keywords_df


########################################






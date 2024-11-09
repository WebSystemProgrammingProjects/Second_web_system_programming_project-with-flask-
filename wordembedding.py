from tqdm import tqdm_notebook as tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from soynlp.tokenizer import RegexTokenizer
import pandas as pd
import numpy as np
from numpy import dot
import math
import warnings
from konlpy.tag import Okt, Hannanum, Kkma, Komoran, Mecab
from numpy.linalg import norm
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import word2vec
import nltk
import urllib.request
import zipfile
from lxml import etree
from tqdm import tqdm
from sklearn.manifold import TSNE
import gensim
import gensim.models as g
from collections import Counter

nltk.download("punkt")
warnings.filterwarnings(action="ignore")

train_data_csv = pd.read_csv("KNOW_2017.csv", low_memory=False)
textColumns = ['bq4_1a', 'bq4_1b', 'bq5_2', 'bq19_1', 'bq30', 'bq31', 'bq32','bq33','bq34']
text = train_data_csv[textColumns]

cols = ['bq4_1a', 'bq4_1b', 'bq5_2', 'bq19_1', 'bq30', 'bq31', 'bq32','bq33','bq34']
text['sentence'] = text[cols].apply(lambda row:''.join(row.values.astype(str)), axis=1)

print(text)
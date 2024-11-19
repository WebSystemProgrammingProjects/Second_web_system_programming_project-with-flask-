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

# nltk.download("punkt")
warnings.filterwarnings(action="ignore")

train_data_csv = pd.read_csv("KNOW_2017.csv", low_memory=False)
textColumns = ['bq4_1a', 'bq4_1b', 'bq5_2', 'bq19_1', 'bq30', 'bq31', 'bq32','bq33','bq34']
text = train_data_csv[textColumns]

cols = ['bq4_1a', 'bq4_1b', 'bq5_2', 'bq19_1', 'bq30', 'bq31', 'bq32','bq33','bq34']
text['sentence'] = text[cols].apply(lambda row:''.join(row.values.astype(str)), axis=1)

tokenizer = RegexTokenizer()

def preprocessing(text):
    text = re.sub('\\n', ' ', text)  # 줄 바꿈 제거
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z, ]', '', text)  # 필요 없는 문자 제거
    return text

sentences = text['sentence'].apply(preprocessing)

tokens = sentences.apply(tokenizer.tokenize)
print(tokens[:])

model = Word2Vec(
    tokens, 
    vector_size=100,  # 벡터 크기
    window=5,         # 문맥 윈도우 크기
    min_count=2,      # 최소 빈도수 (1에서 증가)
    sg=1,            # skip-gram 모델 사용
    workers=4        # 병렬 처리 스레드 수
)

model_name = './iwjoeModel'
model.save(model_name)

loaded_model = Word2Vec.load('./iwjoeModel')

vocab = loaded_model.wv
print(vocab.similarity('컴퓨터', '인공지능'))
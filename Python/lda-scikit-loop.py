import re, nltk, spacy, string
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

import glob
import pandas as pd
import numpy as np

import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

all_files = glob.glob('./*.txt')
i = 0
df = pd.DataFrame(columns = ['text','index'])

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

for infile in sorted(glob.glob('*.txt'), key=numericalSort):
    print('filename: ',infile)
    f = open(infile,encoding="utf8")
    df.loc[i] = [f.read(),i]
    f.close()
    i += 1

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df_clean = pd.DataFrame(df.text.apply(lambda x: clean_text(x)))

nlp = spacy.load('en')
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

df_clean["text_lemmatize"] =  df_clean.apply(lambda x: lemmatizer(x['text']), axis=1)
df_clean['text_lemmatize_clean'] = df_clean['text_lemmatize'].str.replace('-PRON-', '')

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=3,                       
                             stop_words='english',             
                             lowercase=True,                   
                             token_pattern='[a-zA-Z0-9]{3,}',  
                             max_features=5000,          
                            )

data_vectorized = vectorizer.fit_transform(df_clean['text_lemmatize_clean'])

i = 20
doc_topic_prior_set = 0.5 - 0.01*(i/2 + 1)
for x in range(i):
    doc_topic_prior_set += 0.01
    for i in range(1000):
        lda_model = LatentDirichletAllocation(n_components=2,
                                              max_iter=100,
                                              learning_method='online',
                                              learning_offset=50.,
                                              doc_topic_prior=doc_topic_prior_set,
                                              topic_word_prior=0.1)
                                              #,random_state=0)

        lda_output = lda_model.fit_transform(data_vectorized)


        def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=10):
            keywords = np.array(vectorizer.get_feature_names())
            topic_keywords = []
            for topic_weights in lda_model.components_:
                top_keyword_locs = (-topic_weights).argsort()[:n_words]
                topic_keywords.append(keywords.take(top_keyword_locs))
            return topic_keywords

        topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=10)

        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
        df_topic_keywords

        # column names
        topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]

        # index names
        docnames = ["Doc" + str(i) for i in range(len(df))]

        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic['dominant_topic'] = dominant_topic

        df_list = list(df_document_topic['dominant_topic'])
        gr_list = df_list[0:9]
        at_list = df_list[9:14]
        
        gr_mean = sum(gr_list)/len(gr_list)
        at_mean = sum(at_list)/len(at_list)
        
        if abs(gr_mean-at_mean) > 0.7:
            print("GR List: ",gr_list)
            print("AT List: ",at_list)
            print('abs(%f - %f) = %f' % (gr_mean, at_mean, abs(gr_mean-at_mean)))
            
            print(df_topic_keywords.to_string())

            # Log Likelihood: Higher the better
            print("Log Likelihood: ", lda_model.score(data_vectorized))

            # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
            print("Perplexity: ", lda_model.perplexity(data_vectorized))

            # See model parameters
            pprint(lda_model.get_params())

            print(df_document_topic)

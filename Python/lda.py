import pandas as pd
import gensim
import glob
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

all_files = glob.glob('./*.txt')
i = 0
documents = pd.DataFrame(columns = ['text','index'])

for filename in all_files:
    f = open(filename,encoding="utf8")
    documents.loc[i] = [f.read(),i]
    f.close()
    i += 1

def lemmatize_stemming(text):
    wnl = WordNetLemmatizer()
    [(word,tag)]=pos_tag([text])
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    if wntag:
        #print('w: ',word)
        #lemmed = wnl.lemmatize(word, wntag)
        #print('lemmed: ',lemmed)
        return wnl.lemmatize(word, wntag)
    else:
        return word

def preprocess(text):
    result = []
    #print('we are at 3')
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            #print('we are at 4 ',token)
            #print(lemmatize_stemming(token))
            result.append(lemmatize_stemming(token))
    return result

print('we are at 1')
doc_sample = documents[documents['index'] == 1].values[0][0]
print('we are at 2')

doc_sample = documents[documents['index'] == 0].values[0][0]
words = []
for word in doc_sample.split(' '):
    words.append(word)
processed_docs = documents['text'].map(preprocess)
processed_docs[:10]

dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes(no_below=3, no_above=1, keep_n=100)

##count = 0
##for k, v in dictionary.iteritems():
##    print(k, v)
##    count += 1
##    if count > 100:
##        break

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

bow_doc_0 = bow_corpus[0]

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

lda_model = gensim.models.LdaModel(bow_corpus, num_topics=2, id2word=dictionary, eta = 100)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


#print(processed_docs[0])


#for index, score in sorted(lda_model[bow_corpus[0]], key=lambda tup: -1*tup[1]):
#    print("\nScore: {}\t \nGR Topic: {}".format(score, lda_model.print_topic(index, 10)))



#print(processed_docs[12])


#for index, score in sorted(lda_model[bow_corpus[12]], key=lambda tup: -1*tup[1]):
#    print("\nScore: {}\t \nAT Topic: {}".format(score, lda_model.print_topic(index, 10)))












    

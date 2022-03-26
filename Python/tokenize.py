import string, collections, nltk, re, statistics, glob
from nltk.probability import FreqDist
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from readability import Readability
import matplotlib
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
import numpy as np

#paragraph -> array of sentences with punctuation removed
def extract_sents(text):
    #list of sentences
    no_punct = []
    filtered_sent = []
    filtered_sents = []
    text = text.replace('\n','. ')
    sents = sent_tokenize(text)
    for sent in sents:
        no_punct = sent.translate(str.maketrans('','',string.punctuation))
        filtered_sents.append(no_punct)
    return filtered_sents

def extract_words(sents):
    filtered_words=[]
    stop_words=set(stopwords.words("english"))
    regex = re.compile('[^a-zA-Z]')
    lem = WordNetLemmatizer()
    for sent in sents:
        for word, tag in pos_tag(word_tokenize(sent)):
            sent.replace(word,word.strip())
            word = regex.sub('',word)
            if word:
                wntag = tag[0].lower()
                wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                lemma = lem.lemmatize(word, wntag).lower() if wntag else word.lower()
                if lemma not in stop_words:
                    filtered_words.append(lemma)
    return filtered_words

def extract_counts(sents):
    sentence = []
    n_words = 0
    for sent in sents:
        n_words += len(word_tokenize(sent))
    n_chars = sum(len(i) for i in sents)
    n_sents = len(sents)
    return n_chars, n_words, n_sents

def automatic_readability_index(n_chars, n_words, n_sents):
    return 4.71*(n_chars/n_words)+0.5*(n_words/n_sents)-21.43

#create frequency distribution of words
def freqDistWrds(abouts):
    freqList = []
    freqWrds = []
    for about in abouts:
        fdist = FreqDist(about)
        freqList = fdist.most_common(20)
        freqWrds.append(i[0] for i in freqList)
    counter = collections.Counter(x for xs in freqWrds for x in set(xs))
    return counter.most_common(20)

#append each text file to string list
list_of_files = glob.glob('./*.txt')
sentence_holder = []
preserved = []
abouts = []
aris = []
textAbouts = []

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

for infile in sorted(glob.glob('*.txt'), key=numericalSort):
    print('filename: ',infile)
    f = open(infile,encoding="utf8")
    sentence_holder = extract_sents(f.read())
    tokens = word_tokenize(f.read())
    f.close()
    textAbouts.append(nltk.Text(tokens))
    preserved.append('. '.join(sentence_holder))
    abouts.append(extract_words(sentence_holder))
    n_chars, n_words, n_sents = extract_counts(sentence_holder)
    aris.append(automatic_readability_index(n_chars, n_words, n_sents))

wc = []
for file in abouts:
    wc.append(len(file))
    
print('average word count: ',sum(wc)/len(wc))
print('min word count: ',min(wc))
print('max word count: ',max(wc))

grassroots = abouts[0:8]
gr_freq = []
gr_words = []
astroturf = abouts[9:]
at_freq = []
at_words= []

text_gr = textAbouts[0:8]
text_at = textAbouts[9:]

print('Grassroots: ')
print(freqDistWrds(grassroots))
for name, amount in freqDistWrds(grassroots):
    if amount > 1:
        print('\'%s\' is in %s %%' % (name, amount/len(grassroots)*100))
        gr_freq.extend([name,amount/len(grassroots)*100])
        gr_words.append(name)
print('Astroturf: ')
print(freqDistWrds(astroturf))
for name, amount in freqDistWrds(astroturf):
    if amount > 1:
        print('\'%s\' is in %s %%' % (name, amount/len(astroturf)*100))
        at_freq.extend([name,amount/len(astroturf)*100])
        at_words.append(name)

arisData = [sum(aris[0:8])/len(aris[0:8]),sum(aris[9:])/len(aris[9:])]

print('Flesch-Kincaid and SMOG')
ari = []
fk = []
smog = []
for text in preserved:
    r = Readability(text)
    ari_obj = r.ari()
    ari.append(ari_obj.grade_levels)
    fk_obj = r.flesch_kincaid()
    fk.append(fk_obj.grade_level)
    sm_obj = r.smog()
    smog.append(sm_obj.grade_level)

print('ARIs: ',ari)
print('FK ARIs: ',fk)
print('smog ARIs: ',smog)

from matplotlib_venn import venn2, venn2_circles
from matplotlib import pyplot as plt# set up the figure

gr = set(gr_words)
at = set(at_words)

v = venn2([gr,at], ('Grassroots','Astroturf'))

v.get_label_by_id('10').set_text('\n'.join(map(str,gr-at)))
v.get_label_by_id('01').set_text('\n'.join(map(str,at-gr)))
v.get_label_by_id('11').set_text('\n'.join(map(str,gr&at)))# add circle outlines

v.get_patch_by_id('10').set_color('g')
v.get_patch_by_id('10').set_edgecolor('none')
v.get_patch_by_id('10').set_alpha(0.4)

v.get_patch_by_id('01').set_color('mediumblue')
v.get_patch_by_id('01').set_edgecolor('none')
v.get_patch_by_id('01').set_alpha(0.4)

v.get_patch_by_id('11').set_color('skyblue')
v.get_patch_by_id('11').set_edgecolor('none')
v.get_patch_by_id('11').set_alpha(0.4)

label = v.get_label_by_id('10')          
label.set_fontsize(8) 

plt.axis('on')
plt.show()

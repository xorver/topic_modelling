import codecs
import multiprocessing
import re
import collections
import pickle
import copy
import math
import sys
from gensim import corpora, models, similarities

pool = multiprocessing.Pool()

ignored_chars = {'$', '(', ',', '.', ':', ';', '0', '3', '3', '3', '4', '5', '6', '7', '8', '9', '\\', '`', '\'',
                 '+', '-', '*', '/', '<', '>', '^', '%', '=', '?', '!', '[', ']', '{', '}', '_', '\n', '"', '&', '~'}


def base_forms(line):
    tokens = unicode(line, "utf-8").lower().split(", ")
    return tokens


def normalize_text(text):
    text = unicode(text, "utf-8").lower()
    for pattern in ignored_chars:
        text = re.sub(re.escape(pattern), '', text)
    return text.split()


def to_base(args):
    (word_list, base_form) = args
    counter = collections.Counter()
    for word in word_list:
        try:
            counter[base_form[word]] += 1
        except KeyError:
            counter[word] += 1
    return counter

# read odm
base_form = {}
with open("lab8/odm_utf8.txt") as file:
    for line in file.readlines():
        tokens = base_forms(line)
        for el in tokens:
            base_form[el] = tokens[0]

# read pap
with open("lab8/pap.txt") as file:
    text = file.read()
notice_text = re.split(r'#.*', text)
notice_words = map(normalize_text, notice_text)
notice_words = map(lambda words: map(lambda word: base_form.get(word, ''), words), notice_words)
notice_words = map(lambda words: filter(lambda word: word != '', words), notice_words)
notice_counters = map(lambda words: collections.Counter(words), notice_words)

# prepare global counter
global_counter = collections.Counter()
for notice in notice_counters:
    for elem in list(notice):
        global_counter[elem] += 1

# remove singles and >70%
for notice in notice_counters:
    for elem in list(notice):
        if (global_counter[elem] == 1) or (global_counter[elem] > (0.7 * len(notice_counters))):
            del notice[elem]

# prepare BOW
texts = map(lambda x: x.elements(), notice_counters)
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/pap.dict') # store the dictionary, for future reference
dictionary = corpora.Dictionary.load('/tmp/pap.dict')
print(dictionary)

# prepare corpus
corpus = [dictionary.doc2bow(counter.elements()) for counter in notice_counters]
corpora.MmCorpus.serialize('/tmp/pap.mm', corpus) # store to disk, for later use
mm = corpora.MmCorpus('/tmp/pap.mm')
print(mm)

# LSI
lsi = models.lsimodel.LsiModel(corpus=mm, id2word=dictionary, num_topics=100)

# LDA
lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=1)

# Prepare indexes
index_lsi = similarities.MatrixSimilarity(lsi[corpus])
index_lsi.save('/tmp/lsi.index')
index_lsi = similarities.MatrixSimilarity.load('/tmp/lsi.index')
index_lda = similarities.MatrixSimilarity(lda[corpus])
index_lda.save('/tmp/lda.index')
index_lda = similarities.MatrixSimilarity.load('/tmp/lda.index')

# Find
i = 1
n = 3
m = 5

base_bow = dictionary.doc2bow(notice_counters[i].elements())
base_lsi = lsi[base_bow]
sims_lsi = index_lsi[base_lsi]
sims_lsi = sorted(enumerate(sims_lsi), key=lambda item: -item[1])
print("######")
for item in sims_lsi[:n]:
    print notice_text[item[0]]
print("^^^^^^")
sorted_base_lsi = sorted(base_lsi, key=lambda item: -item[1])
for item in sorted_base_lsi[:m]:
    print lsi.print_topic(item[0])


base_lda = lda[base_bow]
sims_lda = index_lda[base_lda]
sims_lda = sorted(enumerate(sims_lda), key=lambda item: -item[1])
print("######")
for item in sims_lda[:n]:
    print notice_text[item[0]]
print("^^^^^^")
sorted_base_lda = sorted(base_lda, key=lambda item: -item[1])
for item in sorted_base_lda[:m]:
    print lda.print_topic(item[0])

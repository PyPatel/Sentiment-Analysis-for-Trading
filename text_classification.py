import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import random

'''
Text Classification can be used for Stock and Politics and other subjects
'''

documents = [(list( movie_reviews.words(fileid), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

'''
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append(list.(movie_reviews.words(fileid)), category)
'''

random.shuffle (documents, random=None)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

print(all_words.most_common(15))

print (all_words["stupid"])


'''
Use ML-Naive Bayes Algorithm to identify possitive & negetive words
'''

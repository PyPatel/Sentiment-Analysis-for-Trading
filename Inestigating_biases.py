import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return  mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents = [(list( movie_reviews.words(fileid), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

print ((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev) , category) for (rev , category) in documents]

#Positive Data example
training = featuresets[:1900]
testing = featuresets[1900:]

# Negative Data example
# training = featuresets[100:]
# testing = featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(training)

# 1st Way
classifier_f = open("naive_bayes.pickle" , "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print ("Naive Bayes Algo Accuracy: ", (nltk.classify.accuracy(classifier, testing))*100)

classifier.show_most_informative_features(20)

# 2nd Way
'''
save_classifier = open("naive_bayes.pickle" , "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
'''

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training)
print ("Multinomial Naive Bayes Algo Accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing))*100)

BRN_classifier = SklearnClassifier(BernoulliNB())
BRN_classifier.train(training)
print ("Bernoulli Naive Bayes Algo Accuracy: ", (nltk.classify.accuracy(BRN_classifier, testing))*100)

GS_classifier = SklearnClassifier(GaussianNB())
GS_classifier.train(training)
print ("Gaussian Naive Bayes Algo Accuracy: ", (nltk.classify.accuracy(GS_classifier, testing))*100)

logistic_classifier = SklearnClassifier(LogisticRegression())
logistic_classifier.train(training)
print ("Logistic Regression Algo Accuracy: ", (nltk.classify.accuracy(logistic_classifier, testing))*100)

#svm_classifier = SklearnClassifier(SVC())
#svm_classifier.train(training)
#print ("SVM Algo Accuracy: ", (nltk.classify.accuracy(GS_classifier, testing))*100)

SGDRegressor_classifier = SklearnClassifier(SGDRegressor())
SGDRegressor_classifier.train(training)
print ("SGDRegressor Algo Accuracy: ", (nltk.classify.accuracy(SGDRegressor_classifier, testing))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training)
print ("LinearSVC Algo Accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training)
print ("NuSVC Algo Accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing))*100)


voted_classifier = VoteClassifier(SGDRegressor_classifier,LinearSVC_classifier,
                                  NuSVC_classifier, MNB_classifier, classifier,
                                  GS_classifier, logistic_classifier, GS_classifier)
print("Voted Classifier Accuracy: " , (nltk.classify.accuracy(voted_classifier, testing))*100)

print("Classification:", voted_classifier.classify(testing[0][0]),"confidence %: ", voted_classifier.confidence(testing[0][0]))

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

syns = wordnet.synset("program")

# Synset
print(syns[0].name())

# Just the word
print(syns[0].lemmas()[0].name())

# Definition
print(syns[0].definition())

# Examples
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synset("good"):
    for l in syn.lemmas():
        #print("l:",l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


# Word Similarity
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
w3 = wordnet.synset("car.n.01")
w4 = wordnet.synset("cat.n.01")

print(w1.wup_similarity(w2))
print(w1.wup_similarity(w3))
print(w1.wup_similarity(w4))


''' Can create Bot that can run a NEWS chanel by scrapping the news from other site and
change the synonyms to create their own article. Do the same thing for the Assignment submission
to cheat the bots.

Google is currently applying this in reverse to find out the Bot Websites which are totally operated
by the Bots.
'''

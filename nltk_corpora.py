import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

print(nltk.__file__)

stop_words = set(stopwords.words("english"))

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample, language='english')

words = word_tokenize(sample, language='english')

filtered_words = [w for w in words if not w in stop_words]
print(tok[5:15], sep=' ', end='n', file=sys.stdout, flush=False)

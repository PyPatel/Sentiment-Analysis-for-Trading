'''
Similar Operation as stemming only the end results will
be real words.
End words will be root words, synonymes to origina; words
'''
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("cows"))
print(lemmatizer.lemmatize("dogy"))

print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better". pos="a"))

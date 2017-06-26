from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Print Stop words
stop_words = set(stopwords.words("english"))
print(stop_words)

example_text = "This is general sentence to just clarify if stop words are working or not. I have some awesome projects coming up"

words = word_tokenize(example_text)

filtered_sentence = []
for w in words:
    for w not in stop_words:
        filtered_sentence.append(w)

#print filtered sentences
print(filtered_sentence)

#print in a line
filtered_sentence1 = [w for w in words if not w in stop_words]

#print filtered sentences
print(filtered_sentence1)

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

example_text = "I was riding the car and going too fast. Suddenly hit the stone and car flipped over"

for w in example_words:
    print(ps.stem(w))

words = word_tokenize(example_text)

for w in words:
    print(ps.stem(w))

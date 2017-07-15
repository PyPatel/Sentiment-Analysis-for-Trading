import nltk
from nltk.corpus import state_union
#PunktSentenceTokenizer is Unsupervised ML tokenizer. We can train it if we want
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
'''
POS TAG List :
Put the words to be tagged here to clean the dataset
Gtting the datat can be anything and do the things we can't do


'''
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = ''' Chunk: {<.*>+}
                                    }<VB.?|IN|DT>+{'''
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            chunked.draw()
#    except Exception as e:
#        print(str(e))

process_content()

from nltk.tokenize import sent_tokenize, word_tokenize


example_text = "How are you Mr. PyPatel. How is your day? Krishna is watching TV. Cool breeze is going on outside. Having fun learning python and nltk."

#print separeted sentances
print (sent_tokenize(example_text))

#print Separeted words
print(word_tokenize(example_text))

#count number of words
count = 0
for i in word_tokenize:
    count+=1

print(count)

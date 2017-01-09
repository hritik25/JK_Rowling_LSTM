import numpy as np
from unidecode import unidecode
import string
import createDataset
# NLTK can be used to split sentences
import nltk.data  


# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Define a function to split a review into parsed sentences
def dataToSentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    rawSentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for rawSentence in rawSentences:
        # If a sentence is empty, skip it
        if len(rawSentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(rawSentence.lower().split())
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def removePunctuation(text):
    for i in xrange(len(text)):
        for j in range(len(text[i])):
            text[i][j] = text[i][j].translate(None, string.punctuation)
    return text


data = []
for i in range(1,8):
    with open('dataset/book_' + str(i)+ '.txt', 'r') as myfile:
        bookText = myfile.read()
        bookText= bookText.decode('utf-8')
        bookText = unidecode(bookText)
        bookText = dataToSentences(bookText, tokenizer)
        bookText = removePunctuation(bookText)
    if i == 1:
        data = bookText
    else:
        data.extend(bookText)


for i in range(len(data)):
    data[i] = ['SENTENCE_START'] + data[i] + ['SENTENCE_END']


vocabSize = 3000
counts, ranks, ranksToWords = createDataset.createDataset(vocabSize)


ranks['SENTENCE_START'] = 3000
ranks['SENTENCE_END'] = 3001
ranksToWords[3000] = 'SENTENCE_START'
ranksToWords[3001] = 'SENTENCE_END'


for i in xrange(len(data)):
    for j in range(len(data[i])):
        if data[i][j] in ranks:
            data[i][j] = ranks[data[i][j]]
        else:
            data[i][j] = ranks['UNK']


# creating the training data
def loadData():
    x_train = []
    y_train = []
    for i in xrange(len(data)):
        x_train.append(data[i][:-1])
        y_train.append(data[i][1:])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    return x_train, y_train
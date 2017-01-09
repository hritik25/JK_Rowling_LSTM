import theano
import theano.tensor as T
import numpy as np
import trainingData
import cPickle
from lasagne.updates import nesterov_momentum

vocabSize = 3002 # 10000 + 2 ( two trivial tokens, SENTENCE_START, SENTENCE_END )
hiddenDim = 128
truncateToTime = -1
NUMBER_OF_EPOCHS = 25 # Number of epochs to train for


#######################
### INITIALIZATIONS ###
#######################

# One recommended approach is to initialize the weights randomly  
# in the interval [-(1/sqrt(n)), 1/(sqrt(n))] where n is the  
# number of incoming connections from the previous layer.

Embeddings = np.random.uniform(-np.sqrt(1./vocabSize), np.sqrt(1./vocabSize), (hiddenDim, vocabSize))
# NOTE : hiddenDim = size of a word embedding 

# Wz has two components, each of dimensions = | hiddenDim x hiddenDim |
Wz = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (2, hiddenDim, hiddenDim))
# Wr has two components, each of dimensions = | hiddenDim x hiddenDim |
Wr = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (2, hiddenDim, hiddenDim))
# W has two components, each of dimensions = | hiddenDim x hiddenDim |
W = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (2, hiddenDim, hiddenDim))
# V will help in projecting the present hidden state to a prediction
V = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (vocabSize, hiddenDim))
# bias has two components, each of dimensions = (hiddenDim,)
bias = np.zeros((2, hiddenDim))
c = np.zeros(vocabSize)

# Creating theano shared variables to store these parameters in GPU memory
Embeddings = theano.shared(np.asarray(Embeddings, dtype = theano.config.floatX))
Wz = theano.shared(np.asarray(Wz, dtype = theano.config.floatX))
Wr = theano.shared(np.asarray(Wr, dtype = theano.config.floatX))
W = theano.shared(np.asarray(W, dtype = theano.config.floatX))
V = theano.shared(np.asarray(V, dtype = theano.config.floatX))
bias = theano.shared(np.asarray(bias, dtype = theano.config.floatX))
c = theano.shared(np.asarray(c, dtype = theano.config.floatX))

# GRU ALGORITHM

# A 2-Layer GRU
# GRU ALGORITHM

# A 2-Layer GRU
def forwardPropagate(wordIndex, ht1Prev, ht2Prev):
    wordEmbedding = Embeddings[:, wordIndex]
    # update gate, zT
    zT1 = T.nnet.sigmoid(T.dot(Wz[0], wordEmbedding) + T.dot(Wz[1], ht1Prev) + bias[0])
    # reset gate, rT
    rT1 = T.nnet.sigmoid(T.dot(Wr[0], wordEmbedding) + T.dot(Wr[1], ht1Prev) + bias[1])
    # candidate ht values, in case they are chosen to be updated
    ht1Candidate = T.tanh(T.dot(W[0], rT1*ht1Prev) + T.dot(W[1], wordEmbedding))
    # ht
    ht1 = (T.ones_like(zT1)-zT1)*ht1Prev + zT1*ht1Candidate
    
    # same for the second layer
    # update gate, zT
    zT2 = T.nnet.sigmoid(T.dot(Wz[2], ht1) + T.dot(Wz[3], ht2Prev) + bias[2])
    # reset gate, rT
    rT2 = T.nnet.sigmoid(T.dot(Wr[2], ht1) + T.dot(Wr[3], ht2Prev) + bias[3])
    # candidate ht values, in case they are chosen to be updated
    ht2Candidate = T.tanh(T.dot(W[2], rT2*ht2Prev) + T.dot(W[3], ht1))
    # ht
    ht2 = T.cast((T.ones_like(zT2)-zT2)*ht2Prev + zT2*ht2Candidate, theano.config.floatX)
    
    # output, dimensions = | vocabSize x 1 |
    # at training time
    # output = T.clip(T.nnet.softmax(T.dot(V, ht2) + c)[0], 1e-6, 1.0-1e-6)
    # at testing time
    output = T.nnet.softmax(T.dot(V, ht2) + c)[0]
    return [output, ht1, ht2]


x = T.ivector()
y = T.ivector()


[output, ht1, ht2], updates = theano.scan( # doc => http://deeplearning.net/software/theano/library/scan.html#theano.scan
        forwardPropagate,
        # sequences - the variables that we want to iterate over. If this  
        # is a matrix, we will be iterating over each row of that matrix.
        sequences = x,
        outputs_info = [None, 
                        dict(initial = T.zeros(hiddenDim)),
                        dict(initial = T.zeros(hiddenDim))],
        truncate_gradient = truncateToTime # -1 indicates all the way back to first step
)


prediction = T.argmax(output, axis = 1)
cost = T.sum(T.nnet.categorical_crossentropy(output,y))
updates = nesterov_momentum(cost, [Embeddings, Wz, Wr, W, V, bias, c], learning_rate = 0.1, momentum = 0.9)
trainLSTMWithSentence = theano.function([x,y], cost, updates = updates)

x_train, y_train = trainingData.loadData()


import time
epochs = 3
start = time.time()
for epoch in xrange(epochs):
    for i in xrange(len(x_train)):
        trainLSTMWithSentence(x_train[i], y_train[i])
        if (i+1)%10000 == 0:
            print '{0} sentences complete.'.format(i+1)
    print 'Epoch {0} complete.'.format(epoch)
end = time.time()
print 'Exection time = '(end - start)/60.0

with open('GRUParametersNew.pkl', 'wb') as file:
    cPickle.dump([Wz.get_value(), Wr.get_value(), W.get_value(), V.get_value(), 
        bias.get_value(), c.get_value(), Embeddings.get_value()], file, protocol = 2)

# Let's test the LSTM
predict = theano.function([x], output)

import createDataset
vocabSize = 3000
counts, ranks, ranksToWords = createDataset.createDataset(vocabSize)

ranks['SENTENCE_START'] = 3000
ranks['SENTENCE_END'] = 3001
ranksToWords[3000] = 'SENTENCE_START'
ranksToWords[3001] = 'SENTENCE_END'

# function to generate the LSTM
def generateSentence(minLength=5):
    # We start the sentence with the start token
    newSentence = [ranks['SENTENCE_START']]
    # Repeat until we get an end token
    while not newSentence[-1] == ranks['SENTENCE_END']:
        nextWordProbabilities = predict(newSentence)[-1]
        samples = np.random.multinomial(1, nextWordProbabilities[1:-2])
        sampledWord = 1+np.argmax(samples)
        newSentence.append(sampledWord)
        print [ranksToWords[i] for i in newSentence]
        # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(newSentence) > 100 or sampledWord == ranks['UNK']:
            return None
    if len(newSentence) < minLength:
        return None
    return newSentence

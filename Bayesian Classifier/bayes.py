import sys
import os
import numpy as np
import nltk
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import heapq
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

config = {'MAX_VOCAB':100, 'MIN_FREQ': 3}


def loadData(path):
  '''
  reads data from the folders
  x_train : [review1, review2, ....., review_n], where each review1 is a list of tokens

  vocabulary is a dictionary: (key: word, value: count)
  '''
  x_train = []
  y_train = []
  vocabulary = []
  x_test = []
  y_test = []
  #Read in the negative reviews for training
  for file in os.listdir(path+'/training_set/neg/'):
      if file.endswith('.txt'):
          with open(path+'/training_set/neg/'+file) as f:
              raw = f.read().strip()
              unstemmed_tokens = tokenizer.tokenize(raw)
              tokens = [ps.stem(word) for word in unstemmed_tokens]
              tokens = [word for word in tokens if word not in stop_words]
              vocabulary+=tokens
              x_train += [tokens]
              y_train += ['neg']

  #Read in the positive reviews for training
  for file in os.listdir(path+'/training_set/pos/'):
      if file.endswith('.txt'):
          with open(path+'/training_set/pos/'+file) as f:
              raw = f.read().strip()
              unstemmed_tokens = tokenizer.tokenize(raw)
              tokens = [ps.stem(word) for word in unstemmed_tokens]
              tokens = [word for word in tokens if word not in stop_words]
              vocabulary+=tokens
              x_train += [tokens]
              y_train += ['pos']

  #Read in the negative reviews for testing
  for file in os.listdir(path+'/test_set/neg/'):
      if file.endswith('.txt'):
          with open(path+'/test_set/neg/'+file) as f:
              raw = f.read().strip()
              unstemmed_tokens = tokenizer.tokenize(raw)
              tokens = [ps.stem(word) for word in unstemmed_tokens]
              tokens = [word for word in tokens if word not in stop_words]
              vocabulary+=tokens
              x_test += [tokens]
              y_test += ['neg']

  #Read in the positive reviews for training
  for file in os.listdir(path+'/test_set/pos/'):
      if file.endswith('.txt'):
          with open(path+'/test_set/pos/'+file) as f:
              raw = f.read().strip()
              unstemmed_tokens = tokenizer.tokenize(raw)
              tokens = [ps.stem(word) for word in unstemmed_tokens]
              tokens = [word for word in tokens if word not in stop_words]
              vocabulary+=tokens
              x_test += [tokens]
              y_test += ['pos']

  # Convert vocabulary list to dictionary
  vocab_dict = {}
  for word in vocabulary:
      if word in vocab_dict.keys():
          vocab_dict[word] = vocab_dict[word]+1
      else:
          vocab_dict[word] = 1

  return x_train, x_test, y_train, y_test, vocab_dict

def getBOWRepresentation(x_train, x_test, vocabulary):
    '''
    converts into Bag of Words representation
    each column is a feature(unique word) from the vocabulary
    x_train_bow : a numpy array with bag of words representation
    '''
    x_train_bow = np.zeros([len(x_train), len(vocabulary)+1])
    for i,review in enumerate(x_train):
        for j,word in enumerate(vocabulary):
            x_train_bow[i][j] = review.count(word)
        x_train_bow[i][-1] = np.sum(len(review)-x_train_bow[i]) #Add UNK vector entry

    x_test_bow = np.zeros([len(x_test), len(vocabulary)+1])
    for i,review in enumerate(x_test):
        for j,word in enumerate(vocabulary):
            x_test_bow[i][j] = review.count(word)
        x_test_bow[i][-1] = np.sum(len(review)-x_test_bow[i]) #Add UNK vector entry

    return np.array(x_train_bow), np.array(x_test_bow)

def naiveBayesMulFeature_train(Xtrain, ytrain):
  thetaPos = np.zeros(config['MAX_VOCAB']+1)
  thetaNeg = np.zeros(config['MAX_VOCAB']+1)
  npN = 0
  nnN = 0
  for i,review in enumerate(Xtrain):
      for j,word in enumerate(review):
          if ytrain[i] == 'pos':
             npN += Xtrain[i,j]
             thetaPos[j] += Xtrain[i,j]
          if ytrain[i] == 'neg':
             nnN += Xtrain[i,j]
             thetaNeg[j] += Xtrain[i,j]
  return (thetaPos+1)/(npN+config['MAX_VOCAB']), (thetaNeg+1)/(nnN+config['MAX_VOCAB'])

def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
  #calculate the probablities, first bringing the prob arrays into log space
  posProbs = np.dot(Xtest, np.log(thetaPos))
  negProbs = np.dot(Xtest, np.log(thetaNeg))
  yPredict = np.where(posProbs > negProbs, 'pos', 'neg')
  correct = 0
  for i,guess in enumerate(yPredict):
      if guess == ytest[i]:
          correct += 1
  return yPredict, correct/len(ytest)

def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
  nbc = MultinomialNB()
  nbc.fit(Xtrain, ytrain)
  return nbc.score(Xtest,ytest)

def naiveBayesBernFeature_train(Xtrain, ytrain):
  thetaPos = np.zeros(config['MAX_VOCAB']+1)
  thetaNeg = np.zeros(config['MAX_VOCAB']+1)
  for i,review in enumerate(Xtrain):
      for j,word in enumerate(review):
          if ytrain[i] == 'pos':
             thetaPos[j] += 1 if Xtrain[i,j] > 0 else 0
          if ytrain[i] == 'neg':
             thetaNeg[j] += 1 if Xtrain[i,j] > 0 else 0
  return (thetaPos+1)/702, (thetaNeg+1)/702

def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
  #convert Xtest to binary array
  Xtest = np.where(Xtest>0,Xtest**0,Xtest)
  NotXtest = np.where(Xtest>0,Xtest-1,Xtest+1)
  #calculate the probablities, first bringing the prob arrays into log space
  posProbs = np.dot(Xtest, np.log(thetaPosTrue)) + np.dot(NotXtest, np.log(1-thetaPosTrue))
  negProbs = np.dot(Xtest, np.log(thetaNegTrue)) + np.dot(NotXtest, np.log(1-thetaNegTrue))
  yPredict = np.where(posProbs > negProbs, 'pos', 'neg')

  correct = 0
  for i,guess in enumerate(yPredict):
      if guess == ytest[i]:
          correct += 1
  return yPredict, correct/len(ytest)


if __name__=="__main__":

    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]



    # read the data and build vocabulary from training data
    XtrainText, XtestText, ytrain, ytest, vocabulary = loadData(textDataSetsDirectoryFullPath)


    # let's look at the vocab
    print("number of unique words: ", len(vocabulary))
    print("the most common 10 words were:", heapq.nlargest(10, vocabulary, key=vocabulary.get))
    print("the least common 10 words were:", heapq.nsmallest(10, vocabulary, key=vocabulary.get))
    vocabulary = dict((word, index) for word, index in vocabulary.items() if vocabulary[word]>=config['MIN_FREQ'] and word in heapq.nlargest(config['MAX_VOCAB'], vocabulary, key=vocabulary.get))
    print("number of unique words in vocabulary: ", len(vocabulary))




    # get BOW representation in the form of numpy arrays
    Xtrain, Xtest = getBOWRepresentation(XtrainText, XtestText, vocabulary=vocabulary.keys())

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)

    print("--------------------")
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)

    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, np.array(ytest), thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")
    print("--------------------")

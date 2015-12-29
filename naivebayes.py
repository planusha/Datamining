__author__ = 'anusha & rohith'
from sklearn.naive_bayes import GaussianNB
import time

class Naivebayes:
    def __init__(self):
        self.trainingMatrix = []
        self.testingMatrix=[]
        self.trainingClass=[]
        self.testTopics=[]
        self.dmatrix = {}
        self.doc_topics = {}

    def set_param(self, dmatrix, doctopic):
        self.dmatrix = dmatrix
        self.doc_topics = doctopic

    def data_matrix(self, train_id, test_id):
        for y in train_id:
            self.trainingMatrix.append(self.dmatrix[y])
            self.trainingClass.append(self.doc_topics[y][0])
        for y in test_id:
            self.testingMatrix.append(self.dmatrix[y])
            self.testTopics.append(self.doc_topics[y][0])

    def classify(self):
        gnb = GaussianNB()
        start_time = time.time()
        gnb.fit(self.trainingMatrix,self.trainingClass)
        print("--- %s naive bayes training time in seconds ---" % (time.time() - start_time))
        i = 0
        cor = 0
        wro = 0
        start_time = time.time()
        for y in self.testingMatrix:
            predictedTopic = gnb.predict(y)
            if self.testTopics[i]!= predictedTopic:
                wro = wro + 1
            else:
                cor = cor + 1
            i = i + 1
        print("--- %s naive bayes testing time in seconds ---" % (time.time() - start_time))
        print "correct prediction:"
        print cor
        print "wrong prediction:"
        print wro

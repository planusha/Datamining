__author__ = 'anusha'
import math
import operator

class Knn:
    def __init__(self):
        self.fv = []
        self.dmatrix = {}
        self.doc_topics = []
        self.test_topics = {}
        self.training = []
        self.testing = []

    def set_param(self, dmatrix, doctopic, fv, train_id, test_id):
        self.dmatrix = dmatrix
        self.doc_topics = doctopic
        self.fv = fv
        self.training = train_id
        self.testing = test_id

    def euclideanDistance(self, fv1, fv2, length):
        distance = 0
        for x in range(length):
            distance += pow((fv1[x] - fv2[x]), 2)
        return math.sqrt(distance)

    def knnClassify(self, x):
        k = 100
        fv_size = len(self.fv)
        distances = []
        fv1 = self.dmatrix[x]
        for y in self.training:
            fv2 = self.dmatrix[y]
            dist = self.euclideanDistance(fv1, fv2, fv_size)
            distances.append((self.doc_topics[y], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbours = []
        for i in range(0,k):
            neighbours.append(distances[i][0])
        return neighbours

    def classify(self):
        check = 0
        wrong = 0
        for x in self.testing:
            neighbours = self.knnClassify(x)
            count = self.most_common(neighbours)
            self.test_topics[x] = count
            if set(count).intersection(self.doc_topics[x]):
                check = check + 1
            else :
                wrong = wrong + 1
        print "correct prediction:"
        print check
        print "wrong prediction:"
        print wrong

    def most_common(self,lst):
        return max(lst,key=lst.count)



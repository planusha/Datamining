__author__ = 'anusha'
import numpy as np
from sklearn.metrics import jaccard_similarity_score

class Jaccard:
    def __init__(self):
        self.dmatrix = {}
        self.doc_ids = []
        self.input_data = []
        self.jaccardsim = {}

    def set_params(self, dmatrix, doc_ids):
        self.dmatrix = dmatrix
        self.doc_ids = doc_ids

    def generateJaccardMatrix(self):
        for doc_id in self.doc_ids:
            self.jaccardsim[doc_id] = {}
        for idx, doc_id1 in enumerate(self.doc_ids):
            fv1 = self.dmatrix[doc_id1]
            fv_size = len(fv1)
            for doc_id2 in self.doc_ids[idx:]:
                fv2 = self.dmatrix[doc_id2]
                sim = self.calcJaccard(fv1, fv2, fv_size)
                #sim = self.calJacSim(fv1, fv2)
                #print sim1
                #print sim
                self.jaccardsim[doc_id1][doc_id2] = sim
                self.jaccardsim[doc_id2][doc_id1] = sim
        return self.jaccardsim

    def calcJaccard(self, fv1, fv2, length):
        sumVal = 0
        uniqueWords = 0
        for x in range(length):
            sumVal = sumVal + (fv1[x]*fv2[x])
            if fv1[x] == 1 or fv2[x] == 1:
                uniqueWords += 1
        if uniqueWords == 0:
            return 0
        else:
            return float(sumVal) / uniqueWords

    def calJacSim(self, fv1, fv2):
        fv1 = np.array(fv1)
        fv2 = np.array(fv2)
        return jaccard_similarity_score(fv1, fv2)

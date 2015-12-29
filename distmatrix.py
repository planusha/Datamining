__author__ = 'anusha'

import math

class distmatrix():
    def __init__(self):
        self.distmatrix = {}
        self.dmatrix = {}
        self.doc_ids = []

    def set_params(self,dmatrix,doc_ids):
        self.dmatrix = dmatrix
        self.doc_ids = doc_ids

    def generate_distmatrix(self):
        for doc_id in self.doc_ids:
            self.distmatrix[doc_id] = {}
        for idx, doc_id1 in enumerate(self.doc_ids):
            fv1 = self.dmatrix[doc_id1]
            fv_size = len(fv1)
            for doc_id2 in self.doc_ids[idx:]:
                fv2 = self.dmatrix[doc_id2]
                dist = self.manhattanDistance(fv1, fv2, fv_size)
                self.distmatrix[doc_id1][doc_id2] = dist
                self.distmatrix[doc_id2][doc_id1] = dist

    def manhattanDistance(self, fv1, fv2, length):
        distance = 0
        for x in range(length):
            distance += math.fabs(fv1[x] - fv2[x])
        return distance

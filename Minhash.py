__author__ = 'anusha'
import random


class Minhash:
    def __init__(self):
        self.dmatrix = {}
        self.doc_ids = []
        self.input_data = []
        self.minhashvals = {}
        self.minhashSimilarity = {}

    def set_params(self, dmatrix, doc_ids):
        self.dmatrix = dmatrix
        self.doc_ids = doc_ids

    def calcMinhash(self, k):
        alist = []
        blist = []
        a = 0
        b = 0
        for doc_id in self.doc_ids:
            self.minhashvals[doc_id] = {}
        for kindex in range(k):
            a = random.randint(1, 512)
            while a in alist:
                a = random.randint(1, 512)
                alist.append(a)
                b = random.randint(1, 512)
                while b in alist:
                    b = random.randint(1, 512)
                blist.append(b)
            for doc_id in self.doc_ids:
                min = 99999
                fv = self.dmatrix[doc_id]
                fv_size = len(fv)
                for x in range(fv_size):
                    hash_val = (a * fv[x] + b) % 557
                    if hash_val < fv_size and hash_val < min and fv[x] == 1:
                        min = hash_val
                self.minhashvals[doc_id][kindex] = min

    def minhashMatrix(self,k):
        for doc_id in self.doc_ids:
            self.minhashSimilarity[doc_id] = {}
        for idx, doc_id1 in enumerate(self.doc_ids):
            fv1 = self.dmatrix[doc_id1]
            fv_size = len(fv1)
            for doc_id2 in self.doc_ids[idx:]:
                self.minhashSimilarity[doc_id1][doc_id2] = self.calcjaccardminhash(self.minhashvals[doc_id1],self.minhashvals[doc_id2],k)
                self.minhashSimilarity[doc_id2][doc_id1] = self.minhashSimilarity[doc_id1][doc_id2]
                #print self.minhashSimilarity[doc_id1][doc_id2]
        return self.minhashSimilarity

    def calcjaccardminhash(self, fv1, fv2, k):
        sumVal = 0
        for x in range(k):
            if fv1[x] == fv2[x]:
                sumVal += 1
        return float(sumVal)/k


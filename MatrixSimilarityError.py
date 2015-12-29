__author__ = 'anusha'

class MatrixSimilarityError:
    def __init__(self):
        self.dmatrix = {}
        self.doc_ids = []
        self.input_data = []

    def set_params(self, dmatrix, doc_ids):
        self.dmatrix = dmatrix
        self.doc_ids = doc_ids

    def calcmeansquare(self, jaccard, minhash, k):
        error = 0;
        n = len(self.doc_ids)
        for idx, doc_id1 in enumerate(self.doc_ids):
            fv1 = self.dmatrix[doc_id1]
            for doc_id2 in self.doc_ids[idx:]:
                error = error + ((jaccard[doc_id1][doc_id2] - minhash[doc_id1][doc_id2]) * (jaccard[doc_id1][doc_id2]- minhash[doc_id1][doc_id2]))
        print "Error"
        print float(error)/(n*(n-3)/4)

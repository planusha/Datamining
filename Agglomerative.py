__author__ = 'anusha'

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import time



class Agglomerative:
    def __init__(self):
        self.dmatrix = {}
        self.distmatrix = {}
        self.doc_ids = []
        self.input_data = []
        self.doc_topics = {}
        self.cluster_entropy = {}
        self.clusters_count = {}
        self.tot_docs = 0

    def set_params(self, dmatrix, doc_ids, doc_topics):
        self.dmatrix = dmatrix
        self.doc_ids = doc_ids
        self.doc_topics = doc_topics

    def get_clusters(self):
        self.get_input();
        X = np.array(self.input_data)
        print("Compute unstructured hierarchical clustering...")
        st = time.time()
        ward = AgglomerativeClustering(n_clusters=80, linkage='complete', affinity='euclidean').fit(X)
        elapsed_time = time.time() - st
        labels = ward.labels_
        print("Elapsed time: %.2fs" % elapsed_time)
        print("Number of points: %i" % labels.size)
        return labels
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        for l in np.unique(labels):
            ax.plot3D(X[labels == l, 0], X[labels == l, 1], X[labels == l, 2],
                      'o', color=plt.cm.jet(float(l) / np.max(labels + 1)))
        plt.title('With connectivity constraints (time %.2fs)' % elapsed_time)
        plt.show()

    def get_input(self):
        for doc_id in self.doc_ids:
            self.input_data.append(self.dmatrix[doc_id])


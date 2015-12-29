__author__ = 'anusha'


from sklearn.cluster import DBSCAN
import numpy
import matplotlib.pyplot as plt
import time


class Dbscan:
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
        self.get_input()
        X = numpy.array(self.input_data)
        start_time = time.time()
        db = DBSCAN(eps=0.2,min_samples=10,metric='euclidean').fit(X)
        print("--- %s DBSCAN time in seconds ---" % (time.time() - start_time))
        #core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        return labels

    def plot_cluster(self, labels, core_samples_mask, X):
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        unique_labels = set(labels)
        colors = plt.cm.Spectral(numpy.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

    def get_input(self):
        for doc_id in self.doc_ids:
            self.input_data.append(self.dmatrix[doc_id])

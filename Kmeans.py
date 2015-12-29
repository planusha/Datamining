__author__ = 'anusha'

import numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time


class Kmeans:
    def __init__(self):
        self.dmatrix = {}
        self.doc_ids = []
        self.input_data = []

    def set_params(self, dmatrix, doc_ids):
        self.dmatrix = dmatrix
        self.doc_ids = doc_ids

    def get_clusters(self):
        self.get_input();
        data = numpy.array(self.input_data)
        kmean_time = time.time()
        reduced_data = PCA(n_components=2).fit_transform(data)
        kmeans = KMeans(n_clusters=80)
        kmeans.fit(reduced_data)
        print("--- %s Kmeans time in seconds ---" % (time.time() - kmean_time))
        labels =kmeans.labels_
        return labels
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
        y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(numpy.c_[xx.ravel(), yy.ravel()])


        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def get_input(self):
        for doc_id in self.doc_ids:
            self.input_data.append(self.dmatrix[doc_id])
import numpy
import math

class Entropy:
    def __init__(self):
        self.doc_ids = []
        self.input_data = []
        self.doc_topics = {}
        self.cluster_entropy = {}
        self.clusters_count = {}
        self.tot_docs = 0

    def set_params(self, doc_ids, doc_topics):
        self.doc_ids = doc_ids
        self.doc_topics = doc_topics

    def preprop_entropy(self, labels):
        clusters_topics = {}
        for x in range(0, len(labels)):
            if labels[x] != -1:
                self.tot_docs += 1
                cluster_id = labels[x]
                doc_id = self.doc_ids[x]
                if cluster_id not in self.clusters_count:
                    self.clusters_count[cluster_id] = 0
                self.clusters_count[cluster_id] += 1
                if cluster_id not in clusters_topics:
                    clusters_topics[cluster_id] = {}
                doc_topic = self.doc_topics[doc_id][0]
                if doc_topic not in clusters_topics[cluster_id]:
                    clusters_topics[cluster_id][doc_topic] = 1
                else:
                    clusters_topics[cluster_id][doc_topic] += 1
        for clus_id in set(labels):
            if clus_id != -1:
                self.cluster_entropy[clus_id] = self.cal_entropy(clusters_topics[clus_id],self.clusters_count[clus_id]);

    def cal_entropy(self,cluster_topics, cluster_count):
        entropy = 0;
        p = cluster_count
        for topic in cluster_topics:
            pi = cluster_topics[topic]
            c = float(pi)/p
            entropy += (-c * math.log(c, 2))
        return entropy

    def cal_total_entropy(self):
        tot_entropy = 0
        D = self.tot_docs
        for clus in self.cluster_entropy:
            m = self.clusters_count[clus]
            tot_entropy += (float(m)/D)*self.cluster_entropy[clus]
        print "entropy"
        print tot_entropy

    def cal_variance(self):
        count = self.clusters_count.values()
        print "number of clusters"
        print len(count)
        print "mean and skew"
        print numpy.mean(count)
        print numpy.var(count)

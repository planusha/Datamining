__author__ = 'anusha'

from dataprep.preprop import DataPreprocess
from dataprep.knn import Knn
from dataprep.naivebayes import Naivebayes
from dataprep.Jaccard import Jaccard
from dataprep.Minhash import Minhash
from dataprep.MatrixSimilarityError import MatrixSimilarityError
import time

class Start():
    def __init__(self):
        self.train_id = []
        self.test_id = []
        self.print_results()

    def print_results(self):
        dp = DataPreprocess()
        #print len(dp.train_set)
        print "Doc# " + str(len(dp.dmatrix1.keys()))
        #print len(dp.feature_vector1)
        k_value = 16
        jaccard_time = time.time()
        js = Jaccard()
        js.set_params(dp.dmatrix1, dp.train_set)
        jaccardMatrix = js.generateJaccardMatrix()
        print("--- %s Jaccard time in seconds ---" % (time.time() - jaccard_time))

        mh = Minhash()
        mh.set_params(dp.dmatrix1, dp.train_set)
        mh.calcMinhash(k_value)
        minHash_time = time.time()
        minhashMatrix = mh.minhashMatrix(k_value)
        print("--- %s MinHash time in seconds ---" % (time.time() - minHash_time))

        error_time = time.time()
        mse = MatrixSimilarityError()
        mse.set_params(dp.dmatrix1,dp.train_set)
        mse.calcmeansquare(jaccardMatrix,minhashMatrix,k_value)
        print("--- %s Error calculation time in seconds ---" % (time.time() - error_time))


        #naive bayes - 256 feature vector call
        #self.call_naivebayes(dp.dmatrix1,dp.topic)
        #self.call_naivebayes(dp.dmatrix2, dp.topic)
        #print "knn - 256"
        #self.call_knn(dp.dmatrix1,dp.topic,dp.feature_vector1)
        #print "knn- 512"
        #self.call_knn(dp.dmatrix2, dp.topic, dp.feature_vector2)
        '''
        ent = Entropy()
        ent.set_params(dp.train_set, dp.topic)
        print "DBSCAN"
        print "256 feature vector"
        db = Dbscan()
        db.set_params(dp.dmatrix1, dp.train_set,dp.topic)
        dbscan_labels = db.get_clusters()
        ent.preprop_entropy(dbscan_labels)
        ent.cal_total_entropy()
        ent.cal_variance()
        print "512 feature vector"
        db = Dbscan()
        db.set_params(dp.dmatrix2, dp.train_set,dp.topic)
        dbscan_labels = db.get_clusters()
        ent.preprop_entropy(dbscan_labels)
        ent.cal_total_entropy()
        ent.cal_variance()
        print "K-Means"
        print "256 feature vector"
        km = Kmeans()
        km.set_params(dp.dmatrix1,dp.train_set)
        kmeans_labels = km.get_clusters()
        ent.preprop_entropy(kmeans_labels)
        ent.cal_total_entropy()
        ent.cal_variance()
        print "512 feature vector"
        km = Kmeans()
        km.set_params(dp.dmatrix2,dp.train_set)
        kmeans_labels = km.get_clusters()
        ent.preprop_entropy(kmeans_labels)
        ent.cal_total_entropy()
        ent.cal_variance()
        print "Hierarchichal Agglomerative"
        print "256 feature vector"
        ag = Agglomerative()
        ag.set_params(dp.dmatrix1, dp.train_set, dp.topic)
        ag_labels = ag.get_clusters()
        ent.preprop_entropy(ag_labels)
        ent.cal_total_entropy()
        ent.cal_variance()
        print "512 feature vector"
        ag = Agglomerative()
        ag.set_params(dp.dmatrix2, dp.train_set , dp.topic)
        ag_labels = ag.get_clusters()
        ent.preprop_entropy(ag_labels)
        ent.cal_total_entropy()
        ent.cal_variance()
        '''

    def split_data(self, ids):
        length = int(len(ids))
        traininglength =int( 0.7 * length)
        self.train_id = ids[:traininglength]
        self.test_id = ids[traininglength:]

    def call_knn(self, dmatrix, doctopic, fv):
        knn = Knn()
        knn.set_param(dmatrix, doctopic, fv, self.train_id, self.test_id)
        start_time = time.time()
        knn.classify()
        print("--- %s knn classify time in seconds ---" % (time.time() - start_time))

    def call_naivebayes(self, dmatrix, doc_topic):
        nb = Naivebayes()
        nb.set_param(dmatrix, doc_topic)
        nb.data_matrix(self.train_id,self.test_id)
        nb.classify()

Start()


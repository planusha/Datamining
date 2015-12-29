__author__ = 'anusha'

import re
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import math


class DataPreprocess:
    def __init__(self):
        path = '/Users/anusha/Documents/studies/DM/assignment1/dataset'
        data_dict = self.data_extraction(path)
        self.train_set = []
        self.test_set = []
        self.topic = {}
        self.doc_tokens = self.stem_stop(data_dict)
        self.no_of_docs = len(data_dict)
        self.doc_word = dict()
        self.tf_idf = dict()
        self.add_to_docword_count()
        self.cal_tf_idf()
        self.feature_vector1 = []
        self.feature_vector2 = []
        self.dmatrix1 = {}
        self.dmatrix2 = {}
        self.generate_feature_vector()
        self.print_data_matrix(data_dict)

    def print_data_matrix(self,data_dict):
        for doc_id in self.train_set:
            self.dmatrix1[doc_id] = []
            self.dmatrix2[doc_id] = []
            for word in self.feature_vector1:
                if word in self.doc_tokens[doc_id]:
                    self.dmatrix1[doc_id].append(1)
                    #self.dmatrix1[doc_id].append(self.doc_tokens[doc_id][word])
                else:
                    self.dmatrix1[doc_id].append(0)
            for word in self.feature_vector2:
                if word in self.doc_tokens[doc_id]:
                    self.dmatrix2[doc_id].append(self.doc_tokens[doc_id][word])
                else:
                    self.dmatrix2[doc_id].append(0)

    def generate_feature_vector(self):
        i = 0
        sorted_tfidf = sorted(self.tf_idf, key=self.tf_idf.get, reverse=True)
        mid = int(len(sorted_tfidf) * 0.01)
        for i in range(mid-128, mid+128):
            self.feature_vector1.append(sorted_tfidf[i])
        for i in range(mid-256, mid + 256):
            self.feature_vector2.append(sorted_tfidf[i])

    def data_extraction(self, path):
        data_dict = {}
        for file in os.listdir(path):
            if file.endswith(".sgm"):
                file = open(path+"/"+file, 'r')
                data = file.read()
                reuters = re.compile('<REUTERS(.*?)</REUTERS>',re.DOTALL).findall(data)
                for x in range(0, 100):
                    class_lables = {}
                    doc_id=re.compile('OLDID="(.*?)"',re.DOTALL).findall(reuters[x])[0]
                    class_lables['TOPICS'] = re.compile('<TOPICS>(.*?)</TOPICS>',re.DOTALL).findall(reuters[x])
                    class_lables['PLACES'] = re.compile('<PLACES>(.*?)</PLACES>',re.DOTALL).findall(reuters[x])
                    class_lables['BODY'] = re.compile('<BODY>(.*?)</BODY>',re.DOTALL).findall(reuters[x])
                    data_dict[int(doc_id)] = class_lables
        return data_dict

    def stem_stop(self, data_dict):
        st = LancasterStemmer()
        doc_tokens = {}
        body_count = 0
        for doc_id in data_dict:
            doc = data_dict[doc_id]
            doc_tokens[doc_id] = {}
            if doc['BODY']:
                body_count = body_count + 1
                body = doc["BODY"][0]
                topics = re.compile('<D>(.*?)</D>',re.DOTALL).findall(doc["TOPICS"][0])
                tokens = word_tokenize(body,language='english')
                if topics:
                    self.topic[doc_id] = topics
                    self.train_set.append(doc_id)
                    tokens = filter(lambda token: token not in topics, tokens)
                else:
                    self.test_set.append(doc_id)
                tokens = filter(lambda token: len(token) > 3, tokens) #removing words of length less than 3
                tokens = filter(lambda token: self.special_match(token), tokens)
                tokens = filter(lambda token: token not in stopwords.words('english'), tokens)# removing stop words
                tokens = map(lambda token: st.stem(token), tokens)#stemming words
                for token in tokens:
                    if token in doc_tokens[doc_id]:
                        doc_tokens[doc_id][token] += 1
                    else:
                        doc_tokens[doc_id][token] = 1
        return doc_tokens

    def special_match(self, token, search=re.compile(r'[^\.a-zA-Z.]').search):
        return not bool(search(token))

    def add_to_docword_count(self):
        for tokens in self.doc_tokens.values():
            for token in tokens:
                if token in self.doc_word:
                    self.doc_word[token] += 1
                else:
                    self.doc_word[token] = 1

    def cal_tf_idf(self):
        for doc_id in self.doc_tokens:
            for token in self.doc_tokens[doc_id]:
                if token not in self.tf_idf:
                    self.tf_idf[token] = self.tf(token, self.doc_tokens[doc_id]) * self.cal_idf(token)
                else:
                    self.tf_idf[token] += self.tf(token, self.doc_tokens[doc_id]) * self.cal_idf(token)

    def tf(self, token, token_values):
        t = max(token_values)
        tf = token_values[token] / token_values[t]
        return tf

    def cal_idf(self, token):
        t = self.no_of_docs / self.doc_word[token]
        if t > 0:
            idf = math.log(t)
        else:
            idf = 1
        return idf

import re
import string

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class Model:
    def __init__(self):
        self.cv, self.clf = self.train()

    def predict(self, data):
        return self.clf.predict(self.pre_process(data))

    def pre_process(self, data):
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([self.count_punct(data)])
        vect = pd.DataFrame(self.cv.transform(data).toarray())
        return pd.concat([body_len, punct, vect], axis=1)

    def train(self):
        data = pd.read_csv("./data/sentiment.tsv", sep="\t")
        data.columns = ["label", "body_text"]

        data["label"] = data["label"].map({"pos": 0, "neg": 1})
        data["tidy_tweet"] = np.vectorize(self.remove_pattern)(
            data["body_text"], "@[\w]*"
        )

        stemmer = PorterStemmer()
        tokenized_tweet = data["tidy_tweet"].apply(lambda x: x.split())
        tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = " ".join(tokenized_tweet[i])

        data["tidy_tweet"] = tokenized_tweet
        data["body_len"] = data["body_text"].apply(lambda x: len(x) - x.count(" "))
        data["punct%"] = data["body_text"].apply(lambda x: self.count_punct(x))

        X = data["tidy_tweet"]
        y = data["label"]

        cv = CountVectorizer()
        X = cv.fit_transform(X)
        X = pd.concat(
            [data["body_len"], data["punct%"], pd.DataFrame(X.toarray())], axis=1
        )
        X = preprocessing.scale(X)

        clf = LogisticRegression(
            C=0.1,
            class_weight=None,
            dual=False,
            fit_intercept=True,
            intercept_scaling=1,
            l1_ratio=None,
            max_iter=100,
            multi_class="auto",
            n_jobs=None,
            penalty="l2",
            random_state=None,
            solver="lbfgs",
            tol=0.0001,
            verbose=0,
            warm_start=False,
        )
        clf.fit(X, y)

        return cv, clf

    @staticmethod
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, "", input_txt)
        return input_txt

    @staticmethod
    def count_punct(txt):
        count = sum([1 for char in txt if char in string.punctuation])
        return round(count / (len(txt) - txt.count(" ")), 3) * 100

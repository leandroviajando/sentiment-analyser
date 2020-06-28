import re
import string
from typing import List

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

TSV_FILE_PATH = "./data/sentiment.tsv"
MODEL = LogisticRegression(
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


class Model:
    def __init__(self):
        self.cv = CountVectorizer()
        self.model = self.__train()

    def predict(self, data: List) -> int:
        return self.model.predict(self.__process(data))

    def __process(self, data: List) -> pd.DataFrame:
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punctuation = pd.DataFrame([self.__count_punctuation(data)])
        vector = pd.DataFrame(self.cv.transform(data).toarray())
        return pd.concat([body_len, punctuation, vector], axis=1)

    def __train(self) -> (CountVectorizer, LogisticRegression):
        training_data = self.__pre_process(pd.read_csv(TSV_FILE_PATH, sep="\t"))
        X = preprocessing.scale(
            pd.concat(
                [
                    training_data["body_len"],
                    training_data["punctuation%"],
                    pd.DataFrame(
                        self.cv.fit_transform(training_data["tidy_tweet"]).toarray()
                    ),
                ],
                axis=1,
            )
        )
        y = training_data["label"]
        return MODEL.fit(X, y)

    def __pre_process(self, training_data: pd.DataFrame) -> pd.DataFrame:
        stemmer = PorterStemmer()
        training_data.columns = ["label", "body_text"]

        training_data["label"] = training_data["label"].map({"pos": 0, "neg": 1})
        training_data["tidy_tweet"] = np.vectorize(self.__remove_pattern)(
            training_data["body_text"], "@[\w]*"
        )

        tokenized_tweet = training_data["tidy_tweet"].apply(lambda x: x.split())
        tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = " ".join(tokenized_tweet[i])

        training_data["tidy_tweet"] = tokenized_tweet
        training_data["body_len"] = training_data["body_text"].apply(
            lambda x: len(x) - x.count(" ")
        )
        training_data["punctuation%"] = training_data["body_text"].apply(
            lambda x: self.__count_punctuation(x)
        )

        return training_data

    @staticmethod
    def __remove_pattern(input_txt: str, pattern: str) -> str:
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, "", input_txt)
        return input_txt

    @staticmethod
    def __count_punctuation(txt: str) -> float:
        count = sum([1 for char in txt if char in string.punctuation])
        return round(count / (len(txt) - txt.count(" ")), 3) * 100

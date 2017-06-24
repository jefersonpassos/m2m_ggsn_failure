#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from gensim.models import Word2Vec

dataset = pd.read_csv("./data/datelo_23_06.csv")

dataset["Status"] = dataset["Status"].apply(lambda x: 1 if x == "FALHA_IDENTIFICADA" else 0)

logs = dataset["LOG"][:5000]
logs = logs.apply(lambda x: x.split(" "))

w2vec = Word2Vec(logs, size = 25, window = 10, min_count = 10, iter = 1, seed = 1)

def log2emb(log):
    wvs = [w2vec.wv[tk] for tk in log if tk in w2vec.wv]
    return list(chain.from_iterable(wvs))

logs = logs.apply(log2emb)

max(map(len, logs))

def normalize_vector(vector):
    while len(vector) < 1400:
        vector.append(0)
    return vector

logs = logs.apply(normalize_vector)

X = {}

for log in logs:
    for idx, val in enumerate(log):
        if not idx in X:
            X[idx] = []
        X[idx].append(val)

X = pd.DataFrame(X)
y = dataset["Status"][:5000]

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#pca = PCA(n_components=50)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)

clf = LinearSVC()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

def predict(log):
    log = log.split(" ")
    log = log2emb(log)
    log = normalize_vector(log)
    return clf.predict(log)

predict("Jun 23 11:15:00 2017  GGS_VOL_01-Re0 (FPC Slot 4, PIC Slot 3) gc43 gtpcd[269]:gc-4/3/0: 8997      <PDP>: <00>: void Pdp::Pd::CreationTimer(void*)() context reached creation time limit, context will be deleted after the current operation is finished imsi=724549000006375, msisdn=5519953270875, nsapi=5, apn=datelo.pc.br, idx=0, s2a_wlan=N, type=21, ip=-, ip_dual=-, sgsn_ip_c=189.40.166.202, sgsn_ip_u=189.40.166.201, teid_data=1234350085, pdp_id=aaaaaaaa987dd720 STATE: [PDN:PDN_ST_INIT(0s)] [0*CREATE:PDP_ST_RADIUS_AUTH(15s):-], first occurrence at 11:15")

dataset["Previsao"] = dataset["LOG"].apply(predict)

dataset[15000:].head(n=50)

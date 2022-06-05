#!/usr/bin/env python
# coding: utf-8

## Library imports

from asyncio.windows_events import NULL
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import difflib

import pickle
from os.path import exists

import timeit

import spacy
# from scipy import spatial

from rank_bm25 import BM25Okapi

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_entities")

nlp2 = spacy.load("en_core_web_md")

stemm = RSLPStemmer()

from flask import Flask
from flask import request

app = Flask(__name__)

# from gensim.models.doc2vec import TaggedDocument
# from gensim.models import Doc2Vec
# from gensim.utils import simple_preprocess

# from random import shuffle

## global Arguments
doc_set = {}
qry_set = {}
rel_set = {}
kw = []
bm25 = NULL

## function for processing data set

def Process_dataset():
    # Processing documents from dataset
    with open('cacm/cacm.all') as f:#CISI/CISI.ALL #cacm/cacm.all
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
    doc_id = ""
    doc_text = ""
    for l in lines:
        if l.startswith(".I"):
            doc_id = int(l.split(" ")[1].strip())-1
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        else:
            doc_text += l.strip()[3:] + " "  # The first 3 characters of a line can be ignored.

    # Processing QUERIES from dataset
    with open('cacm/query.text') as f:#CISI/CISI.QRY #cacm/query.text
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
        qry_id = ""
        for l in lines:
            if l.startswith(".I"):
                qry_id = int(l.split(" ")[1].strip())-1
            elif l.startswith(".W"):
                qry_set[qry_id] = l.strip()[3:]
                qry_id = ""

    # Processing QRELS from dataset
    # with open('CISI/CISI.REL') as f:#CISI/ #cacm/qrels.text
    #     for l in f.readlines():
    #         qry_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0])-1
    #         doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])-1
    #         if qry_id in rel_set:
    #             rel_set[qry_id].append(doc_id)
    #         else:
    #             rel_set[qry_id] = []
    #             rel_set[qry_id].append(doc_id)
    # print(len(doc_set),len(qry_set),len(rel_set))  ####
    with open('cacm/qrels.text') as f:
        for l in f.readlines():
            qry_id = int(l.split(" ")[0])-1
            doc_id = int(l.split(" ")[-1])-1
            if qry_id in rel_set:
                rel_set[qry_id].append(doc_id)
            else:
                rel_set[qry_id] = []
                rel_set[qry_id].append(doc_id)


## function for Preprocessing and tokenize text

def preprocess_text(text):
    token_words = word_tokenize(text)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(stemm.stem(word))
        stem_sentence.append(" ")
    doc = nlp("".join(stem_sentence))
    words = []
    for token in doc:
        if not token.is_punct and not token.is_space and not token.is_stop:
            tk = token.lemma_
            words.append(tk)
    return words


## function for building inverted index using bm25

def building_index(tokenized_corpus):
    bm25_in = BM25Okapi(tokenized_corpus)
    return bm25_in;


## function for return documents most relevant to query

def get_results_bm25(query, bm25):
    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    most_relevant_document = np.argsort(-scores)
    return most_relevant_document


def bulid_keywords():
    for q in list(qry_set.values()):
        for word in q.split():
            if not word in kw:
                kw.append(word)


## function for correct the query

def correct_query(query):
    new_query = " ".join(["".join(difflib.get_close_matches(j,kw,1,cutoff=0.5)) for j in query.split()])
    return new_query

###################################################


Process_dataset()

corpus = list(doc_set.values())
tokenized_corpus = [preprocess_text(doc) for doc in corpus]

if not exists('BM25.index'):
    # build index for Docs
    bm25 = building_index(tokenized_corpus)
    with open('BM25.index', 'wb') as outp:
        pickle.dump(bm25, outp, pickle.HIGHEST_PROTOCOL)
else:
    with open('BM25.index', 'rb') as inp:
        bm25 = pickle.load(inp)




# build key words list
bulid_keywords()


###################################################
###############################
#Evaluating IR Systems Methods#
###############################

num_relevents_docs = {}
num_retrieved_docs = {}

def masked_from_query(qry_id):
    query = qry_set[qry_id]
    rel_docs = []
    if qry_id in rel_set:
        rel_docs = rel_set[qry_id]
    
    most_relevant_document = get_results_bm25(query,bm25)
    # most_relevant_document = rank(query)
    
    masked_relevance_results = np.zeros(most_relevant_document.shape)
    masked_relevance_results[rel_docs] = 1
    sorted_masked_relevance_results = np.take(masked_relevance_results, most_relevant_document)
    num_relevents_docs[qry_id] = np.size(np.atleast_1d(sorted_masked_relevance_results[:10]).nonzero()[0])
    return sorted_masked_relevance_results


## method "one" for MRR@10 Evaluating

def mean_reciprocal_rank(bool_results, k=100):
    bool_results = (np.atleast_1d(r[:k]).nonzero()[0] for r in bool_results)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in bool_results])


## method "tow" for Precision@10 Evaluating

def Precision_At10():
    p_at_10 = {}
    for i in range(0, 10, 1):
        p_at_10[i] = num_relevents_docs[i]/10
    return p_at_10


## method "three" for recall@10 Evaluating

def Recall_At10():
    r_at_10 = {}
    for i in range(0, 10, 1):
        r_at_10[i] = num_relevents_docs[i]/np.size(rel_set[i]) if(np.size(rel_set[i])>0) else 0.
    return r_at_10

###################################################

###################
# Word Embeddings #
###################

# def _embed(tokens):
#     embedding = np.mean(
#         np.array([model[token] for token in tokens if token in model]),
#         axis=0,
#     )
#     q = 0
#     if q==0:
#         print(embedding)
#         q+=1
#     unit_embedding = embedding / (embedding**2).sum()**0.5
#     return unit_embedding

# document_embeddings = np.array([_embed(document) for document in tokenized_corpus]) # (N, E)

# def rank(tokenized_query, mrd):
#     query_embedding = _embed(tokenized_query) # (E,)
#     doc_emb = np.array([document_embeddings[i] for i in mrd])
#     scores = doc_emb.dot(query_embedding)
#     index_rankings = np.argsort(scores)[::-1]
#     return index_rankings, np.sort(scores)[::-1]

# def fucti():
#     for i, doc in enumerate(doc_set.values()):
#         tokens = simple_preprocess(doc)
#         yield TaggedDocument(tokens, [i])
# corpus1 = list(fucti())

# model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(corpus1)
# model.train(corpus1, total_examples=model.corpus_count, epochs=model.epochs)

def _emm(nlp_text):
    emm = np.mean(
    np.array([token.vector for token in nlp_text]),
    axis=0,
    )
    emm_unit = emm / (emm**2).sum()**0.5
    return emm

def rank(query):
    query = " ".join([text for text in preprocess_text(query)])
    query = nlp2(query)
    emm_query = _emm(query)
    results = []
    for doc in docs_emm:
        results.append(doc.dot(emm_query))
    results = np.array(results)
    return results.argsort()[::-1]

def _emm_corpus():
    docs_emm = []
    for doc in tokenized_corpus:
        toks = " ".join([text for text in doc])
        toks = nlp2(toks)
        emm_toks = _emm(toks)
        docs_emm.append(emm_toks)
    return docs_emm

docs_emm = _emm_corpus()
###################################################
###################################################
###############################
#clusturing in k-mean algorithm#
###############################

vectorizer = TfidfVectorizer(stop_words={'english'})
X = vectorizer.fit_transform(doc_set.values())

true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(X)

labels=model.labels_

clst0 = []
clst1 = []
clst2 = []
clst3 = []
clst4 = []
clst5 = []
for key , val in enumerate(labels):
    if val == 0:
        clst0.append(key)
    elif val == 1:
        clst1.append(key)
    elif val == 2:
        clst2.append(key)
    elif val == 3:
        clst3.append(key)
    elif val == 4:
        clst4.append(key)
    elif val == 5:
        clst5.append(key)

def get_results_cluster(query):
    Y = vectorizer.transform([query])
    cluster_id = model.predict(Y)
    clst = []
    if cluster_id == 0:
        clst = clst0
    elif cluster_id == 1:
        clst = clst1
    elif cluster_id == 2:
        clst = clst2
    elif cluster_id == 3:
        clst = clst3
    elif cluster_id == 4:
        clst = clst4
    elif cluster_id == 5:
        clst = clst5
    tokenized_clst = []
    corpus_clst = []
    for i in clst:
        tokenized_clst.append(tokenized_corpus[i])
    for i in clst:
        corpus_clst.append(doc_set[i])
    tokenized_query = preprocess_text(query)
    bm25_clst = BM25Okapi(tokenized_clst)
    scores = bm25_clst.get_scores(tokenized_query)
    most_relevant_document = np.argsort(-scores)
    return most_relevant_document


print("qry1 => " , qry_set[0])
print("rel for qry1 => " , rel_set[0])
###################################################

@app.route('/my-route',methods=['GET'])
def hello_world():
    query = request.args.get('q')
    corr_qry = correct_query(query)
    corriction = "-no change-"
    print(query)

    if query != corr_qry:
        corriction = corr_qry

    start = timeit.default_timer()

    # results = get_results_cluster(query).tolist()
    # results = get_results_bm25(query,bm25).tolist()
    results = rank(query).tolist()
    stop = timeit.default_timer()

    print('Time: ', stop - start)

    json_str = {}

    for id in results:
        json_str[id] = doc_set[id]
    
    returned_array = []
    returned_array.append(json_str)
    returned_array.append(corriction)
    json_string = json.dumps(returned_array)
    return json_string


@app.route('/my-route/evaluating',methods=['GET'])
def evaluating_sys():
    # MMR
    MMR_results = [masked_from_query(qry_id) for qry_id in list(qry_set.keys())]
    mean = mean_reciprocal_rank(MMR_results,10)
    
    # Precision
    Prec = Precision_At10()

    #recall
    recall = Recall_At10()

    jL = []
    jL.append(mean)
    jL.append(Prec)
    jL.append(recall)
    json_eval = json.dumps(jL)
    return json_eval
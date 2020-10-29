# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:57:57 2020
@author: Xuandi Fu
"""

from pytorch_transformers import BertModel, AutoTokenizer
import torch.nn as nn
import torch
from glove import QuestionAnswerPair
from sklearn.manifold import TSNE
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Bert(nn.Module):
    def __init__(self):
        super(Bert,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.BERT = BertModel.from_pretrained("bert-base-uncased")

    def forward(self,inpX):
        inputs = self.tokenizer.encode(inpX)
        inputs = torch.LongTensor(inputs).unsqueeze(0)
        embeddingEach, embeddingOverall = self.BERT(inputs)
        return embeddingOverall

def transform(input, tsne):
    output = tsne.fit_transform(input)
    return output

def fit(input, tsne):
    tsne.fit(input)
    return tsne

def cosine_similarity(x_vec, y_vev):
    """
    :param x_vec: (dim)
    :param y_vev: (n, dim)
    :return:
    """
    y_norm = np.linalg.norm(y_vev, ord=2, axis=1, keepdims=True)
    x_norm = np.linalg.norm(x_vec, ord=2, axis=0, keepdims=True)
    return (y_vev @ x_vec[:, np.newaxis]) / ((y_norm * x_norm) + 1e-12)

def plot(tsne_results, type_names):
    df_subset = pd.DataFrame(tsne_results)

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['type'] = type_names

    num_type = len(set(type_names))
    plt.figure(figsize=(16, 10))

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="type",
        palette=sns.color_palette("hls", num_type),
        data=df_subset,
        legend="full",
        alpha=1
    )
    plt.savefig('subset_plot_10000_each.png')
    return

def get_bert(questions, bert):
    x = np.zeros((len(questions), 768))
    for i, sent in enumerate(questions):
        if len(sent) == 0:
            x[i, :] = np.zeros((1, 768))
            continue
        ques_emb = bert.forward(sent)
        ques_emb = ques_emb.detach().numpy()
        x[i, :] = ques_emb
    return x

if __name__ == "__main__":
    question_path = "../data/OpenEnded_mscoco_train2014_questions.json"
    answer_path = "../data/mscoco_train2014_annotations.json"
    obj = QuestionAnswerPair(question_path, answer_path)
    bert = Bert()
    tsne = TSNE(n_components=2)

    ques_embs = []
    ans_embs = []
    type_names = []
    embeddingeachs = []
    results = []

    for question, answers in obj.iter_question_answer_pairs():
        #print(f"{question}: {type_name}")

        ques_emb = bert.forward(question)
        ques_embs.append(ques_emb.detach().numpy())
        for answer in answers:
            ans_emb = bert.forward(answer)
            ans_embs.append(ans_emb.detach().numpy())

    subset = np.concatenate(ques_embs, axis=0)
    result = transform(subset, tsne)
    # results = np.concatenate(results, axis=0)
    # results.dump('bert.dat')
    # dict = {'embeddings': result, 'type_names':type_names }
    # with open('bert.pickle', 'wb') as handle:
    #     pickle.dump(dict, handle)

    plot(result, type_names)
    print("done!")

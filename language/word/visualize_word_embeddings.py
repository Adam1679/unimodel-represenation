# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:57:57 2020

@author: Anxiang Zhang
"""

from language.glove import Glove, QuestionAnswerPair
import numpy as np
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from string import punctuation
hit = 0
miss = 0
color_map = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
glove_path = "./glove.6B.50d.txt"
answer_path = "./data/mscoco_train2014_annotations.json"
question_path = "./data/OpenEnded_mscoco_train2014_questions.json"
translate_table = dict((ord(char), None) for char in punctuation)
obj = QuestionAnswerPair (answer_path=answer_path, question_path=question_path)
glove_obj = Glove (glove_path)

def __process_sent(sent):
    return sent.lower().translate(translate_table)


def get_sent_embedding(sentence, glove, filter_pos=None):
    global hit, miss
    if filter_pos is None:
        filter_pos = []
    sentence = __process_sent(sentence)
    word_tokens = nltk.word_tokenize (sentence)
    tokens = nltk.pos_tag(word_tokens)
    tokens = [item[0] for item in tokens if item[1] not in filter_pos]
    dim = glove.dim
    sent_emb = np.zeros(dim)
    cnt = 0
    for word in tokens:
        emb = glove.embedding.get(word, None)
        if emb is not None:
            sent_emb += emb
            hit += 1
            cnt += 1
        else:
            miss += 1
    return sent_emb / cnt if cnt > 0 else sent_emb


def all_embeddings(sentences, glove, filter_pos=None):
    x = np.zeros((len(sentences), glove.dim))
    for i, sent in enumerate(sentences):
        x[i, :] = get_sent_embedding(sent, glove, filter_pos)
    return x


def tsne_visualize(X, types, type2name, name):
    """
    :param matrix: np.array with shape (M, dim)
    :param labels:  np.array with shape (M, dim)
    :param types: list of ints
    :return:
    """
    # types = np.array(types)
    X_embedded = TSNE (n_components=2).fit_transform (X)
    f, axes = plt.subplots(1,1,figsize=(10, 10))
    def plot(ax, data, **kwargs):
        return ax.scatter(data[:, 0], data[:, 1], **kwargs)

    color = [color_map[i] for i in types]
    plot(axes, X_embedded, c=color, marker='^', label=type2name[t])

    axes.legend(title_fontsize='small', loc='upper left')
    axes.set_title(f'T-SNE viualization of {name}')
    f.savefig(f"./language.plot.sentence.{name}.png", dpi=180)

def main():
    question_types = [] # integer
    questions = []
    anwser_types = []
    type2id = {}
    id2type = {}
    all_answers = []
    for question, answers, type in obj.iter_question_answer_type_triplets():
        if type not in type2id:
            t = len(type2id)
            type2id[type] = t
            type2id[t] = type

        question_types.append (type2id[type])
        answers = set(answers)  # duplicate answers
        for answer in answers:
            all_answers.append(answer)
            anwser_types.append(type2id[type])

    tsne_visualize(all_embeddings(questions, glove_obj), question_types, id2type, "Questions")
    tsne_visualize(all_embeddings(all_answers, glove_obj), anwser_types, id2type, "Answers")

def main_with_name_entity_filter():
    question_types = [] # integer
    questions = []
    anwser_types = []
    type2id = {}
    id2type = {}
    all_answers = []
    for question, answers, type in obj.iter_question_answer_type_triplets():
        if type not in type2id:
            t = len(type2id)
            type2id[type] = t
            type2id[t] = type

        question_types.append (type2id[type])
        answers = set(answers)  # duplicate answers
        for answer in answers:
            all_answers.append(answer)
            anwser_types.append(type2id[type])

    tsne_visualize(all_embeddings(questions, glove_obj), question_types, id2type, "Questions.filter")
    tsne_visualize(all_embeddings(all_answers, glove_obj), anwser_types, id2type, "Answers.filter")

if __name__ == "__main__":
    main()
    main_with_name_entity_filter()






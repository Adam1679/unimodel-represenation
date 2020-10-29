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
from matplotlib.lines import Line2D
from string import punctuation
hit = 0
miss = 0
color_map = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
cmap = plt.cm.viridis
# cmap = plt.rcParams['image.cmap']
glove_path = "./glove.6B.50d.txt"
answer_path = "./data/mscoco_train2014_annotations.json"
question_path = "./data/OpenEnded_mscoco_train2014_questions.json"
translate_table = dict((ord(char), None) for char in punctuation)
obj = QuestionAnswerPair (answer_path=answer_path, question_path=question_path)
glove_obj = Glove (glove_path)

def __process_sent(sent):
    return sent.lower().translate(translate_table)


def get_sent_embedding(sentence, glove, keep_pos=None):
    global hit, miss
    sentence = __process_sent(sentence)
    word_tokens = nltk.word_tokenize (sentence)
    tokens = nltk.pos_tag(word_tokens)
    tokens = [item[0] for item in tokens if keep_pos is None or item[1] in keep_pos]
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


def all_embeddings(sentences, glove, keep_pos=None):
    # sentences: N strings
    # return: array (N x D)
    x = np.zeros((len(sentences), glove.dim))
    for i, sent in enumerate(sentences):
        x[i, :] = get_sent_embedding(sent, glove, keep_pos)
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
    f, axes = plt.subplots(1, 2,figsize=(30, 15))
    types = np.array(types)
    type1_idx = types < 5
    type2_idx = types >= 5
    X1 = X_embedded[type1_idx, :]
    X2 = X_embedded[type2_idx, :]

    def plot(ax, data, **kwargs):
        return ax.scatter(data[:, 0], data[:, 1], **kwargs)

    plot(axes[0], X1, c=[cmap(color_map[i]) for i in types[type1_idx]], marker='^')
    plot(axes[1], X2, c=[cmap(color_map[i]) for i in types[type2_idx]], marker='^')
    type_ids = np.unique (types)
    legends1 = []
    legends2 = []

    for type_id in type_ids :
        type_name = type2name[type_id]
        line = Line2D ([0], [0], marker='^',
                       color='w',
                       label=type_name,
                       markerfacecolor=cmap(color_map[type_id]),
                       markersize=10)
        if type_id < 5:
            legends1.append(line)
        else:
            legends2.append(line)
    axes[0].legend(handles=legends1, title_fontsize='small', loc='upper left')
    axes[1].legend(handles=legends2, title_fontsize='small', loc='upper left')
    f.savefig(f"./language.plot.sentence.{name}.png", dpi=180)

def main():
    global miss, hit
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
            id2type[t] = type

        question_types.append (type2id[type])
        questions.append(question)
        answers = set(answers)  # duplicate answers
        for answer in answers:
            all_answers.append(answer)
            anwser_types.append(type2id[type])

    tsne_visualize(all_embeddings(questions, glove_obj), question_types, id2type, "Questions")
    print (f"Glove hit/miss in questions: {hit}/{miss}")
    hit = 0
    miss = 0
    tsne_visualize(all_embeddings(all_answers, glove_obj), anwser_types, id2type, "Answers")
    print (f"Glove hit/miss in answers: {hit}/{miss}")


def main_with_name_entity_filter():
    keep_pos = ['NNS', 'NN', 'NNP', 'NNPS', "JJ", 'JJR', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'VB', 'VBD',
                'VBG', 'VBN', 'VBP', 'VBZ']
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
            id2type[t] = type

        question_types.append (type2id[type])
        questions.append(question)
        answers = set(answers)  # duplicate answers
        for answer in answers:
            all_answers.append(answer)
            anwser_types.append(type2id[type])

    tsne_visualize(all_embeddings(questions, glove_obj, keep_pos), question_types, id2type, "Questions.filter")

if __name__ == "__main__":
    main()
    main_with_name_entity_filter()






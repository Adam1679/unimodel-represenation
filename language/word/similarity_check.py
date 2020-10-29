# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:57:57 2020

@author: Anxiang Zhang
"""
from language.glove import Glove, QuestionAnswerPair
from language.word.visualize_word_embeddings import all_embeddings
import numpy as np
from tqdm import tqdm

def cosine_similarity(x_vec, y_vev):
    """
    :param x_vec: (D)
    :param y_vev: (N, D)
    :return: (N) vector
    """
    y_norm = np.linalg.norm(y_vev, ord=2, axis=1, keepdims=True)
    x_norm = np.linalg.norm(x_vec, ord=2, axis=0, keepdims=True)
    return (y_vev @ x_vec[:, np.newaxis]) / ((y_norm * x_norm) + 1e-12)


def get_rank(x_vec, y_vec, y_labels):
    sim = 1 / cosine_similarity(x_vec, y_vec).flatten()
    rank = np.argsort(np.argsort(sim))
    ranks = []
    for label in y_labels:
        ranks.append(rank[label])
    return ranks


def get_mrr(x_vec, y_vec, y_labels):
    ranks = []
    for i, labels in tqdm(enumerate(y_labels)):
        x = x_vec[i, :]
        ranks.extend(get_rank(x, y_vec, labels))

    mrr = [1/(i+1) for i in ranks]
    mrr = sum(mrr) / len(mrr)
    ranks = np.array(ranks)
    top1 = np.mean(ranks == 0)
    top10 = np.mean(ranks < 10)
    top50 = np.mean(ranks < 50)
    return mrr, top1, top10, top50


def get_data(obj, glove_obj, keep_pos=None):
    questions = []
    answer2id = {}
    id2answer = {}
    all_answers = []
    y_labels = []
    for question, answers, type in obj.iter_question_answer_type_triplets ():
        labels = []
        answers = set(answers)  # duplicate answers
        for answer in answers:
            if answer not in answer2id:
                t = len(answer2id)
                answer2id[answer] = t
                id2answer[t] = answer
                all_answers.append (answer)

            labels.append(answer2id[answer])

        y_labels.append(labels)
        questions.append(question)

    return all_embeddings(questions, glove_obj, keep_pos), all_embeddings(all_answers, glove_obj), y_labels


def get_avg_similarity(x_vec, y_vec, y_labels):
    # x_vec: (n, dim)
    # y_vec: (n, dim)
    # reuturn : scalar
    all_sims = []
    for i, labels in enumerate(y_labels):
        sims = cosine_similarity(x_vec[i, :], y_vec[labels]).flatten()
        all_sims.append(sims)
    all_sims = np.concatenate(all_sims)
    return np.nanmean(all_sims)

if __name__ == "__main__":
    glove_path = "./glove.6B.50d.txt"
    answer_path = "./data/mscoco_train2014_annotations.json"
    question_path = "./data/OpenEnded_mscoco_train2014_questions.json"
    obj = QuestionAnswerPair(answer_path=answer_path, question_path=question_path)
    glove_obj = Glove(glove_path)
    # keep_pos = ['NNS', 'NN', 'NNP', 'NNPS', "JJ", 'JJR', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'VB', 'VBD',
    #             'VBG', 'VBN', 'VBP', 'VBZ']
    keep_pos = None
    question_emb, answer_emb, y_labels = get_data(obj, glove_obj, keep_pos)
    # question_emb: (N, D)
    # answer_emb: (M, D)
    # y_labels: length(y_labels) = N. [[0,1,3], [1,2], []]
    sims = get_avg_similarity(question_emb, answer_emb, y_labels)
    mrr, top1, top10, top50 = get_mrr(question_emb, answer_emb, y_labels)
    print(f"MRR {mrr}, Hit@1 {top1:.4%}, Hit@10 {top10:.4%}, Hit@50 {top50:.4%},")
    print(f"Avg CosineSimilarity: {sims:.4f}")
    """
    With NE filter
    MRR 0.000539533095451089, Hit@1 0.0075%, Hit@10 0.0413%, Hit@50 0.1501%,
    Avg CosineSimilarity: 0.4740
    
    Without NE filter: 
    MRR 0.0005426686912533377, Hit@1 0.0038%, Hit@10 0.0338%, Hit@50 0.2176%,
    Avg CosineSimilarity: 0.4479
    """


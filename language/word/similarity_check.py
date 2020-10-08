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
    :param x_vec: (dim)
    :param y_vev: (n, dim)
    :return:
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


def get_data(obj, glove_obj):
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

    return all_embeddings(questions, glove_obj), all_embeddings(all_answers, glove_obj), y_labels

def get_avg_similarity(x_vec, y_vec, y_labels):
    # x_vec: (n, dim)
    # y_vec: (n, dim)
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
    question_emb, answer_emb, y_labels = get_data(obj, glove_obj)
    sims = get_avg_similarity(question_emb, answer_emb, y_labels)
    mrr, top1, top10, top50 = get_mrr(question_emb, answer_emb, y_labels)
    print(f"MRR {mrr}, Hit@1 {top1:.4%}, Hit@10 {top10:.4%}, Hit@50 {top50:.4%},")
    print(f"Avg CosineSimilarity: {sims:.4f}")


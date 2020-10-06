# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:57:57 2020

@author: l
"""
import numpy as np
import json
from dataclasses import dataclass
from collections import defaultdict
class Glove():
    def __init__(self, path):
        self.dim = None
        self.embedding = self.parse(path)
        
    def parse(self, path):
        print("Loading Glove Model")
        f = open(path,'r')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            if self.dim is None:
                self.dim = len(wordEmbedding)
            gloveModel[word] = wordEmbedding
        return gloveModel

@dataclass
class MetaAnswer:
    answer_id: int  # no unique cross questions
    answer: str
    raw_answer: str
    answer_confidence: str # yes/no
    
@dataclass
class MetaAnnotation:
    image_id: str
    answers: list
    confidence: int
    answer_type: str
    question_id: int
    question_type: str
    def __post_init(self):
        answers = []
        for answer in self.answers:
            answers.append(MetaAnswer(**answer))
        self.answers = answers
        
@dataclass
class MetaQuestion:
    image_id: int
    question: str
    question_id: int
    
class QuestionAnswerPair(object):
    def __init__(self, question_path, answer_path):
        self.question_meta = self.parse_question(question_path)
        self.answer_meta = self.parse_answer(answer_path)
        self.question2id = {}
        self.id2question = {}
        self.questionid2type = {}
        self.id2questionmeta = {}
        self.id2answermeta = {}
        self.question_type = {}  # Dict[type_id, type_name]
        self.answer2id = {}
        self.id2answer = {}
        self.questionid2answers = defaultdict(list)
        self.prepare()
        
        
    def parse_answer(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            content = f.read()
        meta_json = json.loads(content)
        return meta_json
    
    def parse_question(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            content = f.read()
        meta_json = json.loads(content)
        return meta_json['questions']
    
    def prepare(self):
        """
        one image -> one question (with unique id)
        (one image, one question) -> 10 anaswers?
        """
        # imageid2questionid = {}
        for part in self.question_meta:
            meta_obj = MetaQuestion(**part)
            self.question2id[meta_obj.question] = meta_obj.question_id
            self.id2question[meta_obj.question_id] = meta_obj.question
            # imageid2questionid[meta_obj.image_id] = meta_obj.question_id
            self.id2questionmeta[meta_obj.question_id] = meta_obj
            
        self.question_type = self.answer_meta['question_types']
        for part in self.answer_meta['annotations']:
            meta_obj = MetaAnnotation(**part)
            self.questionid2type[meta_obj.question_id] = meta_obj.question_type
            for answer in meta_obj.answers:
                obj = MetaAnswer(**answer)
                
                self.questionid2answers[meta_obj.question_id].append(obj.answer)
                if obj.answer not in self.answer2id:
                    tmp = len(self.answer2id)
                    self.answer2id[answer['answer']] = tmp
                    self.id2answer[tmp] = answer['answer']
                    self.id2answermeta[tmp] = obj
                
    def iter_question_answer_pairs(self):
        for question_id, answers in self.questionid2answers.items():
            yield self.id2question[question_id], answers
            
    def iter_question_type_pairs(self):
        for question, question_id in self.question2id.items():
            yield question, self.questionid2type[question_id]
            
if __name__ == "__main__":
    question_path = "D:/OpenEnded_mscoco_train2014_questions.json/OpenEnded_mscoco_train2014_questions.json"
    answer_path = "D:/mscoco_train2014_annotations.json/mscoco_train2014_annotations.json"
    obj = QuestionAnswerPair(question_path, answer_path)
    for question, type_name in obj.iter_question_type_pairs():
        print(f"{question}: {type_name}")
    # print(obj.answer2id)
    # print(f"In total, there are {len(obj.answer2id)} kinds of answers")
            
            
        
        
        
    
    
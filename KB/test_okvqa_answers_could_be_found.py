
import json
from tqdm import tqdm
from stemming.porter2 import stem
from Levenshtein import distance
import numpy as np
import pandas as pd
def get_min_edit_distance(word, vocabs):
    alls = []
    for w in vocabs:
        d = distance(word, w)
        alls.append(d)
    alls = np.array(alls)
    idx = np.argmin(alls)
    return idx, alls[idx]

if __name__ == "__main__":
    file_path = "./KB/all_fact_triples_release.json"
    answer_path = "./data/mscoco_train2014_annotations.json"
    question_path = "./data/OpenEnded_mscoco_train2014_questions.json"
    import sys
    sys.path.append("/Users/adam/Desktop/unimodel-represenation/")
    from language.glove import Glove, QuestionAnswerPair
    f = open(file_path, "r").read()
    json_file = json.loads(f)
    k = list(json_file.keys())[0]
    # print(list(json_file.keys())[:100])
    # dict_keys(['KB', 'e1_label', 'e2_label', 'uri', 'surface', 'dataset', 'sources', 'r', 'context', 'score', 'e1', 'e2'])
    # dict_keys(['KB', 'e1_label', 'e2_label', 'surface', 'isnegated', 'r', 'score', 'e1', 'e2'])
    stem = lambda x: x
    cnt = 0    
    all_entities = set()
    for k, item in json_file.items():
        all_entities.add(item['e1_label'])
        all_entities.add(item['e2_label'])
    print("{} entities".format(len(all_entities)))
    # """ KVQA """
    # with open("./data/KGfacts-CloseWorld.csv") as f:
    #     for line in f:
    #         line = line.strip().split(",")
    #         e1 = line[0].strip()
    #         e2 = line[2].strip()
    #         all_entities.add(e1)
    #         all_entities.add(e2)
    
    # print("{} entities".format(len(all_entities)))
    obj = QuestionAnswerPair(answer_path=answer_path, question_path=question_path)
    hit = 0
    hits = set()
    miss = 0
    for question, answers, type in obj.iter_question_answer_type_triplets ():
        for answer in set(answers):
            if stem(answer) in all_entities:
                hit += 1
                hits.add(stem(answer))
            else:
                miss += 1
                
    print("Hit {}; Miss {}".format(hit, miss))
    
    
    
    
    # s_d = []
    # cnt = 0
    # hit = 0
    # miss = 0
    # for question, answers, type in tqdm(obj.iter_question_answer_type_triplets()):
    #     for answer in set(answers):
    #         answer = stem(answer)
    #         if answer in hits: continue
    #         idx, min_d = get_min_edit_distance(answer, all_entities)
    #         s_d.append(min_d)
    #         if min_d < 2:
    #             hit += 1
    #         else:
    #             miss += 1
        
    # print("Avg d: {}, Hit {}; Miss {}".format(sum(s_d) / len(s_d), hit, miss))

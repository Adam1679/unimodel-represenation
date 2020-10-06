
from pytorch_transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
from glove import QuestionAnswerPair

class Bert(nn.Module):
    def __init__(self):
        super(Bert,self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.BERT = BertModel.from_pretrained("bert-base-uncased")

    def forward(self,inpX):

        inputs = self.tokenizer(inpX, return_tensors="pt")
        embeddingEach, embeddingOverall = self.BERT(inpX)
        self.outputHt = embeddingOverall
        return self.outputHt

if __name__ == "__main__":
    question_path = "OpenEnded_mscoco_train2014_questions.json"
    answer_path = "mscoco_train2014_annotations.json"
    obj = QuestionAnswerPair(question_path, answer_path)
    model = Bert()
    for question, type_name in obj.iter_question_type_pairs():
        print(f"{question}: {type_name}")

        embeding = Bert.forward(question)

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

if __name__ == "__main__":
    question_path = "OpenEnded_mscoco_train2014_questions.json"
    answer_path = "mscoco_train2014_annotations.json"
    obj = QuestionAnswerPair(question_path, answer_path)
    bert = Bert()
    tsne = TSNE(n_components=2)

    embeddings = []
    type_names = []
    embeddingeachs = []
    results = []

    for question, type_name in obj.iter_question_type_pairs():
        #print(f"{question}: {type_name}")

        embedding = bert.forward(question)
        embeddings.append(embedding.detach().numpy())
        type_names.append(type_name)


    subset = np.concatenate(embeddings, axis=0)
    result = transform(subset, tsne)
    # results = np.concatenate(results, axis=0)
    # results.dump('bert.dat')
    # dict = {'embeddings': result, 'type_names':type_names }
    # with open('bert.pickle', 'wb') as handle:
    #     pickle.dump(dict, handle)

    plot(result, type_names)
    print("done!")

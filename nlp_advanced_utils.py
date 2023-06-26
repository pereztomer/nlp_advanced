import numpy as np
from gensim import downloader
import pandas as pd
import ast
import json
import torch


def load_model():
    # WORD_2_VEC_PATH = 'word2vec-google-news-300'
    GLOVE_PATH = 'glove-twitter-200'
    # GLOVE_PATH = 'glove-twitter-25'
    glove = downloader.load(GLOVE_PATH)
    return glove


def embed(model, sen):
    representation = []
    for word in sen:
        word = word.lower()
        if word not in model.key_to_index:
            print(f"The word: [{word}] is not an existing word in the model")
            vec = [0] * 200
        else:
            vec = model[word]
        representation.append([float(x) for x in list(vec)])
    # representation = np.asarray(representation)
    return representation


# def embed_avg(model, sen):
#     representation = []
#     for word in sen:
#         word = word.lower()
#         if word not in model.key_to_index:
#             print(f"The word: [{word}] is not an existing word in the model")
#         else:
#             vec = model[word]
#             representation.append(list(vec))
#
#     representation = np.asarray(representation)
#     representation = np.average(representation, axis=0)
#
#     return representation


# def create_embeddings(csv_path):
#     glove_model = load_model()
#     df = pd.read_csv(csv_path)
#     df['glove_embeddings'] = None
#     for index, row in df.iterrows():
#         current_story = row['preprocessed_text']
#         sen_embedding = embed(model=glove_model, sen=current_story)
#
#
#         df.at[index, 'glove_embeddings'] = sen_embedding
#         # if index == 10:
#         #     break
#
#     df.to_csv('data_team4_embeddings.csv.csv')


def create_embeddings(csv_path):
    glove_model = load_model()
    df = pd.read_csv(csv_path)
    samples = []
    for index, row in df.iterrows():
        current_story = row['preprocessed_text']
        sen_embedding = embed(model=glove_model, sen=current_story)
        sample_dict = row.to_dict()
        sample_dict['glove_embeddings'] = sen_embedding
        samples.append(sample_dict)

    with open("data.json", "w") as outfile:
        outfile.write(json.dumps(samples, indent=4))


from torch.utils.data import DataLoader, Dataset


class CustomDatasetAug(Dataset):
    def __init__(self, json_path, seq_len):
        self.seq_len = seq_len
        self.data = json.load(open(json_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        embeddings = np.array(sample['glove_embeddings'])
        if embeddings.shape[0] < self.seq_len:
            paddings = np.zeros((self.seq_len - embeddings.shape[0], embeddings.shape[1]))
            embeddings = np.concatenate([embeddings, paddings], axis=0)
        elif embeddings.shape[0] > self.seq_len:
            embeddings = embeddings[:self.seq_len]
        label = np.zeros(4)
        label[sample['label'] - 1] = 1  # assumiong labels from 0,1,2,3
        label = label.astype(np.float32)
        return embeddings.astype(np.float32), label


def main():
    from lstm_model import BasicLstm
    model = BasicLstm(embedding_dim=200)
    ds = CustomDatasetAug(json_path='data.json', seq_len=400)
    ds = DataLoader(ds,
                    batch_size=1,
                    shuffle=False)

    for val in ds:
        sample = val[0]  # .to('cuda')
        print(model(sample))
        print('hi')


if __name__ == '__main__':
    main()

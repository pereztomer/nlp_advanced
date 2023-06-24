import numpy as np
from gensim import downloader
import pandas as pd


def load_model():
    # WORD_2_VEC_PATH = 'word2vec-google-news-300'
    # GLOVE_PATH = 'glove-twitter-200'
    GLOVE_PATH = 'glove-twitter-25'
    glove = downloader.load(GLOVE_PATH)
    return glove


def embed(model, sen):
    representation = []
    for word in sen:
        word = word.lower()
        if word not in model.key_to_index:
            print(f"The word: [{word}] is not an existing word in the model")
            vec = np.zeros(200)
        else:
            vec = model[word]
        representation = representation + list(vec)
    representation = np.asarray(representation)
    return representation


def embed_avg(model, sen):
    representation = []
    for word in sen:
        word = word.lower()
        if word not in model.key_to_index:
            print(f"The word: [{word}] is not an existing word in the model")
        else:
            vec = model[word]
            representation.append(list(vec))

    representation = np.asarray(representation)
    representation = np.average(representation, axis=0)

    return representation


def create_embeddings(csv_path):
    glove_model = load_model()
    df = pd.read_csv(csv_path)
    df['glove_embeddings'] = None
    for index, row in df.iterrows():
        current_story = row['preprocessed_text']
        sen_embedding = embed_avg(model=glove_model, sen=current_story)
        df.at[index, 'glove_embeddings'] = sen_embedding
        # if index == 10:
        #     break

    df.to_csv('data_team4_embeddings.csv.csv')

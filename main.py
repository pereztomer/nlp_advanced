import numpy as np
from gensim import downloader
from nlp_advanced_utils import create_embeddings


def main():
    csv_path = 'data_team4 - emberassing400.csv'
    create_embeddings(csv_path)


if __name__ == '__main__':
    main()

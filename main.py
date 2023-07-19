import numpy as np
from gensim import downloader
from nlp_advanced_utils import create_embeddings


def main():
    csv_path = 'embrassing_400_by_annotators.csv'
    create_embeddings(csv_path)


if __name__ == '__main__':
    main()

import pandas as pd
import csv

from generate_word2vec_embeddings import Comments

SIZE=100
EPOCHS=5
MIN_COUNT=10
WORKERS=40
DATA_PATH='data/comments_processed.csv'

from gensim.models.fasttext import FastText as FT_gensim

ft = FT_gensim(
    size=SIZE,
    min_count=MIN_COUNT,
    workers=WORKERS
    )

ft.build_vocab(sentences=Comments(DATA_PATH))

ft.train(
    sentences=Comments(DATA_PATH),
    epochs=EPOCHS,
    total_examples=ft.corpus_count, 
    total_words=ft.corpus_total_words
    )

ft.save('data/embeddings/fastext_{}_{}min_{}epochs'.format(SIZE, MIN_COUNT, EPOCHS))
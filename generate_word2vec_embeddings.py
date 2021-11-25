import pandas as pd
import csv

SIZE=100
EPOCHS=5
MIN_COUNT=10
WORKERS=40
DATA_PATH='data/comments_processed.csv'

class Comments(object):
    def __init__(self, filepath):
        self.filepath = filepath
        
    def __iter__(self):        
        with open(self.filepath, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                yield row[-1].split()

from gensim.models import Word2Vec

comments = Comments(DATA_PATH)
w2v = Word2Vec(
    comments,
    size=SIZE,
    iter=EPOCHS,
    min_count=MIN_COUNT,
    workers=WORKERS
    )

w2v.save('data/embeddings/w2v_{}_{}min_{}epochs'.format(SIZE, MIN_COUNT, EPOCHS))
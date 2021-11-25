import csv
import string
from nltk.corpus import stopwords
import gensim


MIN_COUNT = 10
BIGRAM = False
TRIGRAM = False
DATA_PATH = 'data/all_text_clean.csv'
DICT_PATH = 'data/dict_{}mn'.format(MIN_COUNT)


import logging
logging.basicConfig(filename='data/logs/dict_{}mn.log'.format(MIN_COUNT), format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Comments(object):
    def __init__(self, filepath):
        self.filepath = filepath
        
    def __iter__(self):        
        with open(self.filepath, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                if len(row) == 0:
                    continue
                yield row[-1].split()


comments = Comments(DATA_PATH)

# bigram = gensim.models.Phrases(sentences=comments, min_count=MIN_COUNT, threshold=100)
# trigram = gensim.models.Phrases(sentences=bigram[all_words], min_count=5, threshold=100)

# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# all_words = [bigram_mod[doc] for doc in all_words]
id2word = gensim.corpora.Dictionary(comments)

# prune dictonary
id2word.filter_extremes(no_below=MIN_COUNT)

id2word.save(DICT_PATH)
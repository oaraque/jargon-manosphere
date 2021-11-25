import csv
import gensim

N_TOPICS=110
PASSES=5
WORKERS=10
MIN_COUNT=10
DATA_PATH='data/all_text_clean.csv'

import logging
logging.basicConfig(filename='data/logs/lda_model_{}topics_{}passes.log'.format(N_TOPICS, PASSES), format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class Comments(object):
    def __init__(self, filepath):
        self.filepath = filepath
        
    def __iter__(self):        
        with open(self.filepath, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                if len(row) == 0:
                    continue
                yield row[-1].split()


id2word = gensim.corpora.Dictionary.load('data/dict_{}mn'.format(MIN_COUNT))

memory_estimation = (8 * len(id2word) * N_TOPICS * 3) / 2**30 
logger.info('MEMORY ESTIMATION: {}'.format(memory_estimation))

comments = Comments(DATA_PATH)
corpus = [id2word.doc2bow(text) for text in comments]

lda_model = gensim.models.ldamulticore.LdaMulticore(
    corpus=corpus,
    id2word=id2word,
    num_topics=N_TOPICS,
    random_state=42,
    chunksize=50000,
    passes=PASSES,
    per_word_topics=True,
    workers=WORKERS,
)

lda_model.save('data/lda_model_{}topics_{}passes'.format(N_TOPICS, PASSES))

import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tqdm.pandas(desc="progress")

import spacy
nlp = spacy.load("en", disable=['tagger', 'parser', 'ner'])

# cleaning utilities
stop_words = set(stopwords.words('english'))

noise = set(string.punctuation) | set(['\n'])
noise = {ord(c): None for c in noise}

def preprocess_text(text):
    return ' '.join([token.text.lower() for token in  nlp.tokenizer(text)])

def clean(text):
    text = text.translate(noise).split(' ')
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

def read_csv(file_path, name):
    logger.info('Reading %s ...' % name)
    df = pd.read_csv(file_path, header=0)
    logger.info('Done. Shape: %s' % str(df.shape))
    return df


def preprocess_step(comments, links):
    #logger.info('Reading comments...')
    #comments = pd.read_csv('data/sql_dump/comments.csv', header=0)
    #logger.info('Done. Shape: %s' % str(comments.shape))

    # number of comments that are null
    null_comments = comments['body'].isnull()
    logger.debug(null_comments.sum())

    comments['body_processed'] = comments['body'][~null_comments].progress_apply(preprocess_text)

    #logger.info('Reading links...')
    #links = pd.read_csv('data/sql_dump/links.csv')
    #logger.info('Done. Shape: %s' % str(links.shape))

    links['title_processed'] = links['title'].progress_apply(preprocess_text)
    # links that have no test
    null_links = links['self_text'].isnull()
    links.loc[~null_links, 'self_text_processed']  = links[~null_links]['self_text'].progress_apply(preprocess_text)

    return comments, links 

def export_preprocess(comments, links):
    null_comments = comments['body'].isnull()
    null_links = links['self_text'].isnull()
    # export comments and links separately
    logger.info('Export processed DFs')
    comments.to_csv('data/comments_processed.csv', header=True, index=None)
    links.to_csv('data/links_processed.csv', header=True, index=None)

    # export just text
    logger.info('Export just processed text')
    all_text_processed = np.concatenate(
        (comments[~null_comments]['body_processed'].values, 
        links['title_processed'].values,
        links[~null_links]['self_text_processed'].values), axis=0)
    pd.Series(all_text_processed).to_csv('data/all_text_processed.csv', header=False, index=None)

def clean_step(comments, links):
    null_comments = comments['body'].isnull()
    null_links = links['self_text'].isnull()

    logger.info('Cleaning text...')
    comments.loc[~null_comments, 'body_clean'] = comments[~null_comments]['body_processed'].progress_apply(clean)
    links.loc[:, 'title_clean'] = links['title_processed'].progress_apply(clean)
    links.loc[~null_links, 'self_text_clean'] = links[~null_links]['self_text_processed'].progress_apply(clean)
    logger.info('Done')
    return comments, links

def export_clean(comments, links):
    null_comments = comments['body'].isnull()
    null_links = links['self_text'].isnull()
    # export comments and links
    logger.info('Export cleaned DFs')
    comments.drop(['body', 'body_processed'], axis=1).to_csv('data/comments_clean.csv', header=True, index=None)
    links.drop(['title', 'title_processed', 'self_text', 'self_text_processed'], axis=1).to_csv('data/links_clean.csv', header=True, index=None)

    # export just text
    logger.info('Export just clean text')
    all_text_processed = np.concatenate(
        (comments[~null_comments]['body_clean'].values, 
        links['title_clean'].values,
        links[~null_links]['self_text_clean'].values), axis=0)
    pd.Series(all_text_processed).to_csv('data/all_text_clean.csv', header=False, index=None)

if __name__ == '__main__':

    if not os.path.exists('data/comments_processed.csv') \
    or not os.path.exists('data/links_processed.csv'):
        logger.info('Entering preprocessing')
        comments = read_csv('data/sql_dump/comments.csv', 'comments')
        links = read_csv('data/sql_dump/links.csv', 'links')
        comments, links = preprocess_step(comments, links)
        export_preprocess(comments, links)
    
    if not os.path.exists('data/comments_clean.csv') \
        or not os.path.exists('data/links_clean.csv'):
        logger.info('Entering cleaning')
        comments = read_csv('data/comments_processed.csv', 'comments')
        links = read_csv('data/links_processed.csv', 'links')

        comments, links = clean_step(comments, links)
        export_clean(comments, links)
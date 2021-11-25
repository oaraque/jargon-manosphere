import logging
import wordfreq
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from tqdm import tqdm
import os
import inspect
import time
import datetime
import json
import gensim
from gensim.models.fasttext import FastText as FT_gensim
from collections import Counter
from itertools import chain
from Levenshtein import distance
from sklearn.manifold import TSNE

import neologism_discovery_config
config = dict()

logger = logging.getLogger(__name__)

noise = set(string.punctuation)
noise = {ord(c): None for c in noise}

# preprocess words
digits = set(string.digits)
punctuation = set(string.punctuation)
ascii_letters = set(string.ascii_lowercase)

def check_word_is_clean(word):
    word = set(word)
    is_clean = (len(word & punctuation) == 0) and \
        (len(word & punctuation) == 0) and \
        (len(word - ascii_letters) == 0)
    return is_clean

def check_topic_by_string(word, lda_model, minimum_probability=None):
    try:
        topics = lda_model.get_term_topics(word, minimum_probability=minimum_probability)
    except IndexError:
        topics = []
    return topics

def similar_term(wo, wt):
    wo = wo.translate(noise)
    wt = wt.translate(noise)
    if distance(wo, wt) < config['levenshtein_distance_threshold']:
        return True
    if (wo in wt) or (wt in wo):
        return True
    return False

def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M")

def text_size_frequency_based(freq):
    base = 8
    variable = 0.01 * freq
    if variable > 13:
        variable = 10
    return int(base + variable)

def load_data(custom_config=None):

    set_config(custom_config)

    # LDA stuff
    id2word = gensim.corpora.Dictionary.load(config['LDA_DICT_PATH'])
    # this selection has done with coherence in mind
    lda = gensim.models.ldamodel.LdaModel.load(config['LDA_MODEL_PATH']) 

    # comments = Comments('data/all_text_clean.csv')
    # corpus = [id2word.doc2bow(text) for text in comments]

    # Embeddings stuff
    ft = FT_gensim.load(config['EMBEDDINGS_MODEL_PATH'])

    # All texts data
    comments = pd.read_csv(config['COMMENTS_PROCESSED_PATH'], header=0)
    null_comments = comments['body'].isnull()
    comments = comments[~null_comments]


    links = pd.read_csv(config['LINKS_PROCESSED_PATH'], header=0)
    null_links = links['self_text'].isnull()
    links = links[~null_links]

    all_texts = pd.concat((comments[['id', 'body_processed', 'created_utc']], links[['id', 'self_text_processed', 'created_utc']]), axis=0, sort=False)
    all_texts = all_texts[~all_texts['id'].isnull()]
    all_texts['text'] = all_texts['body_processed'].fillna(all_texts['self_text_processed'])
    all_texts.drop(['body_processed', 'self_text_processed'], axis=1, inplace=True)

    count = Counter(
        list(chain.from_iterable(all_texts['text'].str.split()))
    )

    return id2word, lda, ft, all_texts, count

def filter_dict_freq(count, report):
    all_words_count = pd.DataFrame.from_dict(count, orient='index')
    report['results']['terms_dataset'] = len(all_words_count)
    logger.info('No. of terms in dataset: {:,}'.format(len(all_words_count)))
    all_words = all_words_count[all_words_count[0] > config['frequency_threshold']].index.values
    all_words = set(all_words)
    report['results']['frequency_filtered_terms'] = len(all_words)
    logger.info('No. of terms with count > {}: {:,}'.format(config['frequency_threshold'], len(all_words)))

    # remove not clean words
    all_words = {w for w in all_words if check_word_is_clean(w)}
    report['results']['clean_filtered_terms'] = len(all_words)
    logger.info("No. of clean terms: {:,}".format(len(all_words)))

    # filter by generic dictionary
    general_words = set(wordfreq.get_frequency_dict('en', wordlist='large').keys())
    report['results']['general_words'] = len(general_words)
    logger.info("No. of 'general' words: {:,}".format(len(general_words)))
    all_words = all_words - general_words
    report['results']['dictionary_filtered_terms'] = len(all_words)
    logger.info("No. of terms after generic dictionary filtering: {:,}".format(len(all_words)))
    return all_words, general_words, report

def override_candidates(all_words, method, size=100, seeds=None):
    if method == 'random':
        return np.random.choice(list(all_words), size=size)
    elif method == 'seeds':
        return seeds


def lda_topic_distribution(lda, all_words, report, show_plots=False):
    topics = []
    for neo_word in all_words:
        topics.extend(check_topic_by_string(
            neo_word, 
            lda,
            minimum_probability=config['lda_minimum_probability']))
        
    covered_topics = {t[0] for t in topics}
    report['results']['lda_topics'] = lda.num_topics
    report['results']['lda_covered_topics'] = len(covered_topics)
    logger.info("No. of  topics covered: {} / {} = {:.2f}% ".format(
        len(covered_topics),
        lda.num_topics,
        (len(covered_topics) / lda.num_topics) * 100
    ))

    # print topic distribution
    if not show_plots:
        return report

    lda_count = {i: 0 for i in range(lda.num_topics)}
    lda_count.update(Counter([t[0] for t in topics]))

    labels, values = zip(*lda_count.items())
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.show()

    return report

def embedding_expansion(count, all_words, general_words, ft, report):
    neo_dict = dict()
    neo_V = []
    for target in all_words:
        derived_words = [word for word, distance in ft.wv.similar_by_word(target, topn=config['n_neighbours']) if not similar_term(target, word)]
        neo_V.extend(derived_words)
        neo_dict.update({w: target for w in derived_words})

    words_V = neo_V
    words_V.extend(all_words)
    words_V = pd.DataFrame(columns=['word'], data=words_V)
    neo_freqs = [count[w] for w in words_V['word'].values]
    words_V['freq'] = neo_freqs
    words_V['seed'] = words_V['word'].apply(lambda w: w in all_words)
    words_V['parent'] = words_V['word'].apply(lambda w: neo_dict.get(w, None))
    words_V['dictionary'] = words_V['word'].apply(lambda w: w in general_words)

    # in the expansion duplicates can appear
    words_V.drop_duplicates(['word',], inplace=True)

    # filter by frequency
    words_V = words_V[words_V['freq'] > config['frequency_threshold']]
    # remove not clean words
    words_V = words_V[words_V['word'].apply(lambda w: check_word_is_clean(w))]

    expansion_factor = (words_V.shape[0] - len(all_words)) / len(all_words)
    report['results']['embedding_expansion_factor'] = expansion_factor
    logger.info("Expansion factor of the embeding expansion: {:.2f}".format(expansion_factor))

    dictionary_words = words_V['dictionary'].value_counts() 
    dict_words = int(dictionary_words.loc[True])
    no_dict_words = int(dictionary_words.loc[False])
    report['results']['embedding_expansion_dict_words'] = dict_words
    report['results']['embedding_expansion_no_dict_words'] = no_dict_words
    logger.info('No. of words that are inside a generic dictionary after expansion: {:,} ({:.2f}%)'.format(dict_words, 100 * dict_words / words_V.shape[0]))
    logger.info('No. of words that are outside a generic dictionary after expansion: {:,} ({:.2f}%)'.format(no_dict_words, 100 * no_dict_words / words_V.shape[0]))

    new_neo_candidates = len(set(words_V[~words_V['dictionary']]['word']) - set(all_words))
    report['results']['embedding_new_neo_candidates'] = new_neo_candidates
    logger.info("No. of candidates found after embedding expasion: {:,}".format(new_neo_candidates))

    return words_V, report

def filter_emb_levenshtein(words_V, target, similarity_threshold, ft):
    ws = [w for w, sim in ft.similar_by_vector(target) if sim > similarity_threshold]
    candidates = words_V[words_V['word'].isin(ws)].sort_values('freq', ascending=False).drop_duplicates('word')
    if candidates.shape[0] == 0:
        return None, None
    candidates_v = candidates['word'].values
    distances_matrix = np.empty((len(candidates_v), len(candidates_v)))
    for i, w_i in enumerate(candidates_v):
        for j, w_j in enumerate(candidates_v):
            similar = similar_term(w_i, w_j)
            distances_matrix[i,j] = similar
    
    indxs = candidates[candidates['word'].isin(candidates_v[distances_matrix.sum(axis=0) > 1])].index
    # if word are distant, indxs is empty
    if len(indxs) == 0:
        return None, None
    
    retained = indxs[0]
    filtered = indxs[1:].values
    return retained, filtered


def filter_distances(words_V, current_words, filtered_words, ft):
    #global words_V
    #nonlocal current_words
    #global filtered_words
    target = select_word(current_words)
    # target = 'plasty'
    retained, filtered = check_word(words_V, target, ft)
    if (not retained is None) and (not filtered is None):
        remove_words = words_V['word'].loc[filtered].values
        remove_words = np.append(remove_words, [target,], axis=0)
        filtered_words.update(target)
        words_V = words_V[~words_V.index.isin(filtered)]
    else:
        remove_words = np.array([target,])
    current_words = np.setdiff1d(current_words, remove_words)
    return words_V, current_words, filtered_words
    
def select_word(l):
    return np.random.choice(l, size=1)[0]  
    
def check_word(words_V, target, ft):
    return filter_emb_levenshtein(words_V, target, config['similarity_threshold'], ft)
    
def do_distance_filtering(words_V, current_words, filtered_words, ft):

    words_V, current_words, filtered_words = filter_distances(
        words_V, current_words, filtered_words, ft)

    current_before = current_words.shape[0]

    with tqdm(total=current_words.shape[0]) as pbar:
        while current_words.shape[0] > 0:
            words_V, current_words, filtered_words = filter_distances(words_V,
                current_words, filtered_words, ft)
            pbar.update(current_before - current_words.shape[0])
            current_before = current_words.shape[0]

    return words_V

def draw_embedding_plot(count, words_V, ft):
    logger.info('Extracting neologisms embeddings')

    words_V = words_V.reset_index()

    neo_V = [ft.wv[w] for w in words_V['word'].values]
    neo_V = np.array(neo_V)

    logger.info('Doing TSNE dimensionality reduction')
    neo_V_reduced = TSNE(n_components=2, random_state=42).fit_transform(neo_V)

    logger.info('Generating 2D plot')
    plt.figure(figsize=(30,30))
    plt.scatter(neo_V_reduced[:, 0], neo_V_reduced[:, 1], c=None, marker=',', linewidth=0.3 )
    for i, row in words_V.iterrows():
        w = row['word']
        size = text_size_frequency_based(count[w])
        if words_V.loc[i, 'parent'] is None:
            color = 'darkgreen'
        elif not words_V.loc[i, 'dictionary']:
            color = 'darkorange'
        else:
            color = None
        plt.annotate(w, (neo_V_reduced[i, 0], neo_V_reduced[i, 1]), size=size, color=color)

    export_file  = os.path.join(config['SAVE_PATH'], 'neologisms_embedding_plot_{}.pdf'.format(get_timestamp())) 
    logger.info('Exporting embedding plot to {}'.format(export_file))
    plt.savefig(export_file, bbox_to_anchor=True)

    # save embeddings to file
    emb_save = pd.DataFrame(data=neo_V)
    export_file  = os.path.join(config['SAVE_PATH'], 'neologisms_embeddings_{}.tsv'.format(get_timestamp())) 
    logger.info('Saving embedding vectors to {}'.format(export_file))
    emb_save.to_csv(export_file, sep='\t', header=None, index=None)
    # save embedding words
    export_file  = os.path.join(config['SAVE_PATH'], 'neologisms_embeddings_words_{}.txt'.format(get_timestamp())) 
    logger.info('Saving vector words to {}'.format(export_file))
    words_V['word'].to_csv(export_file, index=None)
    

def save(words_V):
    export_file = os.path.join(config['SAVE_PATH'], 'neologism_expansion_{}.csv'.format(get_timestamp()))
    logger.info('Exporting neologism list to {}'.format(export_file))
    words_V.to_csv(export_file, header=True, index=False) 

def save_report(report):
    export_file = os.path.join(config['SAVE_PATH'], 'neologism_expansion_report_{}.json'.format(get_timestamp()))
    logger.info('Exporting report to {}'.format(export_file))
    with open(export_file, 'w') as f:
        json.dump(report, f)

def write_dir():
    if os.path.exists(config['SAVE_PATH']):
        return
    os.makedirs(config['SAVE_PATH'])


def set_config(custom_config=None):
    global config

    if custom_config is None:
        default_config = {member[0]: member[1] for member in inspect.getmembers(neologism_discovery_config) if not member[0].startswith('_')}
        config = default_config
    else:
        config = custom_config

    logger.debug(config)

def run(custom_config=None, do_embedding_filtering=True, show_plots=False, loaded=False, id2word=None, lda=None, ft=None,
    all_texts=None, count=None, override=False, override_method='random', override_seeds=None, override_size=100, embedding_plot=False, do_save=False):
    '''
    Run the neologism discovery method.
    loaded -> if True, other data values must be provided
    '''

    set_config(custom_config)
    write_dir()

    report = {
        'parameters': {k: v for k, v in config.items()},
        'results': {}
    } 

    # load data
    if not loaded:
        id2word, lda, ft, all_texts, count = load_data(custom_config=custom_config)
    else:
        id2word, lda, ft, all_texts, count = id2word, lda, ft, all_texts, count

    # frequency and dictionary based filtering
    all_words, general_words, report = filter_dict_freq(count, report)
    # we can override candidates if desired
    if override:
        all_words = override_candidates(all_words, method=override_method, size=override_size, seeds=override_seeds)
        report['results']['override_size'] = override_size
        report['results']['override_method'] = override_method

    # lda topic distribution
    report = lda_topic_distribution(lda, all_words, report, show_plots)

    # embedding expansion
    words_V, report =  embedding_expansion(count, all_words, general_words, ft, report)
    pre_emb_filtering = words_V[~words_V['dictionary']].shape[0]
    report['results']['embedding_pre_filtering'] = pre_emb_filtering
    logger.info('Total number of candidates found: {:,}'.format(pre_emb_filtering))

    # embedding filtering 
    if do_embedding_filtering:
        filtered_words = set([])
        current_words = words_V[~words_V['dictionary']]['word'].unique()
        words_V = do_distance_filtering(words_V, current_words, filtered_words, ft)

        post_emb_filtering = words_V[~words_V['dictionary']].shape[0]
        report['results']['embedding_post_filtering'] = post_emb_filtering 
        logger.info('Embedding filtering has removed {:,} terms'.format(pre_emb_filtering - post_emb_filtering))
        logger.info('Total number of candidates after embedding filtering: {:,}'.format(post_emb_filtering))

    # lda topic distribution
    report = lda_topic_distribution(lda, all_words, report, show_plots)

    # 2D plot
    if embedding_plot:
        draw_embedding_plot(count, words_V, ft)

    # save
    if do_save:
        save(words_V)
    save_report(report)

    return words_V, report

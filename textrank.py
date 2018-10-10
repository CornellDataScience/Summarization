"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang
This is an implementation of the textrank algorithm, an adaptation of the
pagerank algorithm for keyword extraction and extractive summarization."""

import spacy
import numpy as np
import en_core_web_sm
import networkx as nx
import tensorflow as tf
import tensorflow_hub as hub
from spacy.attrs import POS, LEMMA, IS_STOP
from sklearn.metrics.pairwise import cosine_similarity

nlp = en_core_web_sm.load()

####################### KEYWORD EXTRACTION #######################################
def get_lemma_nodes(doc_arr, POS_tags=[83, 91, 95]):#Adjective, Noun, Proper noun
    """Returns the unique lemma hashes of tokens with a part of speech in POS_tags
    and that aren't stop words"""
    tokens_of_POS = doc_arr[np.isin(doc_arr[:,1],POS_tags)]
    tokens_wo_stopwords = tokens_of_POS[tokens_of_POS[:,2] == 0]
    nodes = np.unique(tokens_wo_stopwords)
    return nodes


def generate_keyword_edge_weights(nodes, doc, n=6):
    """Returns the cooccurence weights of each given node.
    Scores are weighted such that the closer the word is the higher the score
    and are constrained to be within [n] tokens of eachother.
    TODO: make more efficient"""
    n_nodes = len(nodes)
    inverse_ix = {key: ix for ix, key in enumerate(nodes.tolist())}
    co_occur_weights = np.zeros((n_nodes,n_nodes))
    for ix, tok in enumerate(doc[n:]):
        node_ix = inverse_ix.get(tok.lemma, -1)
        if node_ix != -1: #If the token is a node
            window = doc[max(0, ix-n):min(len(doc), ix+n)] #All tokens within n tokens
            for window_ix, tok2 in enumerate(window):
                node2_ix = inverse_ix.get(tok2.lemma, -1)
                if node2_ix != -1 and window_ix != n: #If the second token is a different node
                    co_occur_weights[node_ix, node2_ix] += 1/(abs(window_ix-n))
    return co_occur_weights


def construct_lemma_graph(doc):
    '''Returns the graph used to calculate pagerank on.
    doc : spaCy Doc object
    returns: networkx Graph'''
    doc_arr = doc.to_array([LEMMA, POS, IS_STOP])
    nodes = get_lemma_nodes(doc_arr)
    edge_weights = generate_keyword_edge_weights(nodes, doc)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    nonzero_x, nonzero_y = np.nonzero(edge_weights)
    w_edges = [(nodes[n1], nodes[n2], max([edge_weights[n1,n2], edge_weights[n2,n1]])) \
               for n1, n2 in zip(nonzero_x, nonzero_y )]
    G.add_weighted_edges_from(w_edges)
    return G


def keyword_extraction(text, n_words):
    '''Returns the most significant [n_words] as determined by the textrank algorithm.'''
    doc = nlp(text)
    G = construct_lemma_graph(doc)
    pr = nx.pagerank_numpy(G)
    top_ranked = sorted(list(pr.items()), key=lambda x: x[1],reverse=True)[:n_words]
    return [doc.vocab.strings[node[0]] for node in top_ranked]
##################################################################################

###############################SUMMARIZATION####################################
def get_sentence_nodes(sens):
    '''Returns a list of sentence embeddings based on the google universal sentence encoder.
    sens : list of strings corresponding to each sentence'''
    with tf.Graph().as_default():
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        embed = hub.Module(module_url)
        embeddings = embed(sens)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            vecs = sess.run(embeddings)

    return vecs

def get_sentence_edge_weights(sentence_vecs):
    '''Returns a dict of the form d[sentence1][sentence2] = cosine_sim(sentence1, sentence2)
        where sentence1 always comes before sentence2.
        sentence_vecs: list of sentence embeddings'''
    weight_dict = {}
    for ix, sent in enumerate(sentence_vecs[:-1]):
        weight_dict[ix] = {}
        for jx, sent2 in enumerate(sentence_vecs[ix+1:]):
            weight_dict[ix][ix+jx] = cosine_similarity([sent], [sent2])
    return weight_dict

def construct_sentence_graph(sens):
    '''Returns a networkx graph with nodes corresponding to the index of the sentence
    in the text and edges with weight equal to the cosine similarity between the two
    sentence embeddings.
    sens: a list of strings'''
    sentence_mats = get_sentence_nodes(sens)
    weight_dict = get_sentence_edge_weights(sentence_mats)
    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(sentence_mats))])
    edge_weights = [(ix, jx, weight) for ix, inner_dict in weight_dict.items()
                    for jx, weight in inner_dict.items() ]
    G.add_weighted_edges_from(edge_weights)
    return G

def summarize(text, n_sentences):
    '''Returns the [n_sentences] from [text] that had the highest pagerank score.
    text: string
    n_sentences: int '''
    doc = nlp(text)
    sens = [sen.text for sen in list(doc.sents)]
    G = construct_sentence_graph(sens)
    pr = nx.pagerank_numpy(G)
    top_ranked = sorted(list(pr.items()), key=lambda x: x[1],reverse=True)[:n_sentences]
    ordered_sens = sorted(top_ranked, key=lambda x: x[0])
    return [sens[node[0]] for node in ordered_sens]


def get_summary(text):
    """
    Generate and return summary from text based off of how many sentences it contains, 
    to be called outside of module
    """
    num_words = text.count(" ")
    num_sentences = text.count(".")
    keywords = keyword_extraction(text, 5)
#    summary = summarize(text, max(1, num_sentences//10))[0]
    return keywords, None


if __name__ == '__main__':
    print("enter text below:")
    test_text = input()
    print("ok")
    info = get_summary(test_text)

    print("##########EXTRACTING KEYWORDS############")
    print(info)

#    print("##########GENERATING SUMMARY############")
#    print(info[1])
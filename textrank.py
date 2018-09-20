"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

This is an implementation of the textrank algorithm, an adaptation of the
pagerank algorithm for keyword extraction and extractive summarization."""

import spacy
import numpy as np
import en_core_web_sm
import networkx as nx
from spacy.attrs import POS, LEMMA, IS_STOP
from spacy.lang.en.stop_words import STOP_WORDS


def get_lemma_nodes(doc_arr, POS_tags=[83, 91, 95]):#Adjective, Noun, Proper noun
    """Returns the unique lemma hashes of tokens with a part of speech in POS_tags
    and that aren't stop words"""
    tokens_of_POS = doc_arr[np.isin(doc_arr[:,1],POS_tags)]
    tokens_wo_stopwords = tokens_of_POS[tokens_of_POS[:,2] == 0]
    nodes = np.unique(tokens_wo_stopwords[:,0])
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
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    G = construct_lemma_graph(doc)
    pr = nx.pagerank_numpy(G)
    top_ranked = sorted(list(pr.items()), key=lambda x: x[1],reverse=True)[:5]
    return [doc.vocab.strings[node[0]] for node in top_ranked]


if __name__ == '__main__':
    test_text = """There is a large body of work in extractive text summarization, due to its easier nature. Perhaps the most famous approach is called textrank which is an adaptation of the pagerank algorithm that is used to identify the most important sentences in a text. Work in abstractive text summarization has increased recently due to the rise of deep learning and its success in text generation as well as some success in reading comprehension.
	Knowledge graphs have also been around for some time now, with Google having a large knowledge graph of over 70 billion nodes. Recent advances in this area have been in generating these graphs directly from unstructured text. Deep learning again has provided some tools that have been helpful in progressing the constructing of these graphs.
	At the intersection of knowledge graphs and summarization is an area that is sometimes called information cartography. The graphs generated in this area are much less granular than knowledge graphs, and instead read more like summaries. Additionally, the nodes are events or concepts and instead of edges being labeled, paths are labeled, with some sort of story line that ties the nodes in that path together. These objects are more useful for summarizing bigger more complex subjects with a significant amount of text material."""
    print(keyword_extraction(test_text, 6))

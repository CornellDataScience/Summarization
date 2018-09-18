"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

This is an implementation of the textrank algorithm, an adaptation of the
pagerank algorithm for keyword extraction and extractive summarization."""

import spacy
import en_core_web_sm
import networkx as nx
from spacy.attrs import POS, LEMMA


def get_nodes(doc_arr):
    nouns_and_adj_codes = [83, 91, 95] #Adjective, Noun, Proper noun
    nouns_and_adj = doc_arr[np.isin(doc_arr[:,1],nouns_and_adj_codes)]
    nodes = np.unique(nouns_and_adj[:,0])
    return nodes


def generate_edge_weights(nodes, doc, n=6):
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
                    co_occur_weights[node_ix, node2_ix] += 1/(abs(ix-n))
    return co_occur_weights


def construct_graph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    w_edges = [(nodes[n1], nodes[n2], max([edge_weights[n1,n2], edge_weights[n2,n1]])) \
               for n1, n2 in zip(nonzero_x, nonzero_y )]
    G.add_weighted_edges_from(w_edges)
    return G


def keyword_extraction(text, n_words):
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    doc_arr = doc.to_array([LEMMA, POS])
    nodes = get_nodes(doc_arr)
    edge_weights = generate_edge_weights(nodes, doc)
    G = construct_graph(nodes, edge_weights)
    pr = nx.pagerank_numpy(G)
    top_ranked = sorted(list(pr.items()), key=lambda x: x[1],reverse=True)[:5]
    return [doc.vocab.strings[node[0]] for node in top_ranked]

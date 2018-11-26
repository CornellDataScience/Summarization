# -*- coding: utf-8 -*-
"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

This file contains the functions for testing and evaluating the performance
of the KG construction pipeline and summarization procedure."""

import pickle
import networkx as nx
import matplotlib.pyplot as plt


def unpickle_kg(dir):
    G = pickle.load(open(dir + 'graph.p', 'rb'))
    sum_G = pickle.load(open(dir + 'sum_graph.p', 'rb'))
    relations = pickle.load(open(dir + 'relations.p', 'rb'))
    entities = pickle.load(open(dir + 'entities.p', 'rb'))
    return G, sum_G, relations, entities

def display_graph(dir=''):
    G, sum_G, relations, entities = unpickle_kg(dir)
    ents = list(entities.items())
    sorted_ents = sorted(ents, key=lambda d: d[1]['doc_apps'])
    for k, d in sorted_ents:
        print('ID:',k, '  Apps:',d['doc_apps'], '   Text:', d['text'])
    for e in G.edges():
        print(e)
    plt.figure()
    nx.draw_networkx(G)
    plt.show()


display_graph('Data/trump_russia/kg/')

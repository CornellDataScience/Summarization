# -*- coding: utf-8 -*-
"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

This file contains the functions for testing and evaluating the performance
of the KG construction pipeline and summarization procedure."""

import pickle
import networkx as nx
import matplotlib.pyplot as plt
#from main import summary

def unpickle_kg(dir):
    G = nx.read_gpickle(open(dir + 'graph.p', 'rb'))
    sum_G = nx.read_gpickle(open(dir + 'sum_graph.p', 'rb'))
    relations = pickle.load(open(dir + 'relations.p', 'rb'))
    entities = pickle.load(open(dir + 'entities.p', 'rb'))
    return G, sum_G, relations, entities

def display_graph(dir=''):
    G, sum_G, relations, entities = unpickle_kg(dir)
    ents = list(entities.items())
    sorted_ents = sorted(ents, key=lambda d: d[1]['doc_apps'])
    rels = nx.get_edge_attributes(G,'relationship')
    print(rels)
    for k, d in sorted_ents:
        print('ID:',k, '  Apps:',d['doc_apps'], '   Text:', d['text'])
    for e in G.edges():
        print('Edge:', entities[e[0]]['text'], '--', relations[rels[(e[0],e[1],0)]],
                '-->', entities[e[1]]['text'])
    plt.figure()
    nx.draw_networkx(G)
    plt.show()

def test(test_name):
    text = ""
    with open("test/" + test_name+".txt", 'r', encoding='utf-8') as f:
        text = f.read()

    result = summary(text, test_name)
    f = open("test/" + test_name+"_result.txt", "w")
    f.write(result[0]+ "\n")
    f.write(result[1]+ "\n")
    f.write(result[2]+ "\n")

if __name__ == '__main__':
    display_graph('Data/trump_russia/kg/')
    #test("test1")

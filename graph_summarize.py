# -*- coding: utf-8 -*-
"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

summarizes knowledge graphs """

import networkx as nx
from collections import deque


def bfs_expansion(G, source, depth_limit=None):
    G_ = G.to_undirected()
    visited = [source]
    if depth_limit is None:
        depth_limit = len(G)
    q = deque([(source, 0, nx.neighbors(G_, source))])
    while q:
        parent, depth_now, children = q[0]
        for child in children:
            if child not in visited:
                visited.append(child)
                if depth_now < depth_limit:
                    q.append((child, depth_now + 1, nx.neighbors(G_, child)))

        q.popleft()
    return visited


def greedy_summarize(G, k, c, degree_weight):
    selected = []
    dom = []
    #print("g1: "  + str(G.nodes[1]))
    largest_degree_vs = sorted(G.nodes.keys(), key=lambda i: G.nodes[i]['weight'] + G.degree()[i] * degree_weight) [-k:]
    for i in largest_degree_vs:
        depth = int(G.nodes[i]['weight'] * c)
        selected += bfs_expansion(G, i, depth_limit=depth)
        dom += nx.dominating_set(G, i)

    selected = set(selected).intersection(set(dom))
    return nx.subgraph(G, selected)

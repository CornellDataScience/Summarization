"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

This is an implementation of the textrank algorithm, an adaptation of the
pagerank algorithm for keyword extraction and extractive summarization."""

import editdistance
import itertools
import networkx as nx
import nltk


def build_graph(nodes):
    """Return a networkx graph instance.
    :param nodes: List of hashtables that represent the nodes of a graph.
    """
    #initialize graph
    graph = nx.Graph()  
    graph.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    #add edges weighted by Levenshtein distance
    for pair in nodePairs:
        levDistance = editdistance.eval(pair[0], pair[1])
        graph.add_edge(pair[0], pair[1], weight=levDistance)

    return graph


def extract_sentences(text, clean_sentences=False, language='english'):
    """Return a paragraph formatted summary of the source text.
    :param text: A string.
    """
    sent_detector = nltk.data.load('tokenizers/punkt/'+language+'.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    graph = build_graph(sentence_tokens)

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    #list of most important sentences first
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)

    return sentences

def get_summary(text, numSentences=1):
    '''
    Return summary of text in numSentences sentences (default: 1)
    '''
    sentences = extract_sentences(text)

    if numSentences > len(sentences) or numSentences == 1: summary = sentences[0]
    
    else: 
        summary = ' '.join(sentences[:numSentences])
        summary_words = summary.split()
        dot_indices = [idx for idx, word in enumerate(summary_words) if word.find('.') != -1]
        if dot_indices:
            last_dot = max(dot_indices) + 1
            summary = ' '.join(summary_words[:last_dot])
        else:
            summary = ' '.join(summary_words)

    print(summary)
    return summary

if __name__ == "__main__":
    print("###########Input text here:############")
    text = input()
    print("#########Generated Summary:############")
    get_summary(text)

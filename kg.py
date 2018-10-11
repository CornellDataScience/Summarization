"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

This script generates knowledge graphs from unstructured text which has been
inspired by the T2KG paper."""

import spacy
import numpy as np
import networkx as nx
import en_core_web_sm
from spacy import displacy
from collections import Counter

class Entity:
    doc_appearances = set()
    type = None
    aliases = set()

    def __init__(self, name, index):
        self.name = name
        self.aliases.append(name)
        self.index = index

    def merge(self, other_ent):
        self.aliases = self.aliases.union(other_ent.aliases)
        self.doc_appearances = self.doc_appearances.union(other_ent.doc_appearances)


class KG:
    self.graph = None
    self.entities = {}
    self.keys = 0
    self.docs = []
    def __init__(self):
        pass

    def entity_detection(self):
        '''Compile a list of entities from a collection of documents.
        doc_dict - {document index: spacy Doc object}'''
        for ix, doc in self.doc_dict.items():
            for ent in doc.ents:
                new_ent = Entity(ent.text, self.keys)
                new_ent.type = ent.label_
                new_ent.doc_appearances.append((ix, ent.start, ent.end))
                self.entities.append(new_ent)
                self.keys += 1


    def coreference_detection():
        pass

    def triple_extraction():
        pass

    def graph_construction():
        pass

def create_kg(text_dict):
    nlp = en_core_web_sm.load()
    docs = {ix:nlp(doc) for ix, doc in text_dict.items()}

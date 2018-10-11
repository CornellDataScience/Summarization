"""
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

This script generates knowledge graphs from unstructured text which has been
inspired by the T2KG paper."""

import spacy
import pickle
import numpy as np
import networkx as nx
import en_core_web_sm
from spacy import displacy
from collections import Counter

class Entity:
    '''The entity class is for storing named entities with the KG. These will
    eventually be used to create the nodes of the KG.'''
    def __init__(self, name, index):
        #The plain text representation
        self.name = name
        #A set containing all plain text representations of the same entity
        self.aliases = {name}
        #The unique integer index value given by the KG to the entity
        self.index = index
        #List of appearences in the document corpus formatted as (doc_ix, token_ix_start, token_ix_end)
        self.doc_appearances = []
        #Spacy entity type
        self.type = None

    def merge(self, other_ent):
        '''Updates the entity to contain the aliases and appearences of another entity
        object representing that same underlying entity.'''
        self.aliases = self.aliases.union(other_ent.aliases)
        self.doc_appearances += other_ent.doc_appearances


class KG:
    '''The KG class is for maintaining all of the data associated with the
    knowledge graph as well as the various procedures for construction.'''
    def __init__(self):
        #Stores the final constructed graph
        self.graph = nx.MultiDiGraph()
        # {entity id: entity object}
        self.entities = {}
        # {entity name: entity ix}
        self.name_to_ix = {}
        # {entity ix: entity name}
        self.ix_to_name = {}
        # {document id: spacy doc object}
        self.doc_dict = {}
        # The number of unique index values given out for entities
        self.keys = 0

    def entity_detection(self):
        '''Compile a list of entities from a collection of documents.
        doc_dict - {document index: spacy Doc object}'''
        for ix, doc in self.doc_dict.items():
            for ent in doc.ents: #For all entities in all documents
                if ent.text in self.name_to_ix:
                    #If ent already exists, add appearence
                    self.entities[self.name_to_ix[ent.text]].doc_appearances.append((ix, ent.start, ent.end))
                else:
                    #Else create new entity and update KG data fields
                    new_ent = Entity(ent.text, self.keys)
                    new_ent.type = ent.label_
                    new_ent.doc_appearances.append((ix, ent.start, ent.end))
                    self.entities[self.keys] = new_ent
                    self.name_to_ix[ent.text] = self.keys
                    self.ix_to_name[self.keys] = ent.text
                    self.keys += 1


    def coreference_detection():
        pass

    def triple_extraction():
        pass

    def graph_construction():
        pass


text = '''The first step in solving any problem is admitting there is one. But a new report from the US Government Accountability Office finds that the Department of Defense remains in denial about cybersecurity threats to its weapons systems.

Specifically, the report concludes that almost all weapons that the DoD tested between 2012 and 2017 have “mission critical” cyber vulnerabilities. “Using relatively simple tools and techniques, testers were able to take control of systems and largely operate undetected, due in part to basic issues such as poor password management and unencrypted communications,” the report states. And yet, perhaps more alarmingly, the officials who oversee those systems appeared dismissive of the results.

The GAO released its report Tuesday, in response to a request from the Senate Armed Services Committee ahead of a planned $1.66 trillion in spending by the Defense Department to develop its current weapons systems. Subtitled "DoD Just Beginning to Grapple with Scale of Vulnerabilities," the report finds that the department "likely has an entire generation of systems that were designed and built without adequately considering cybersecurity." Neither Armed Services Committee chairman James Inhofe nor ranking member Jack Reed responded to requests for comment.

The GAO based its report on penetration tests the DoD itself undertook, as well as interviews with officials at various DoD offices. Its findings should be a wakeup call for the Defense Department, which the GAO describes as only now beginning to grapple with the importance of cybersecurity, and the scale of vulnerabilities in its weapons systems.

“I will say that the GAO can be prone to cyber hyperbole, but unless their sampling or methodology were way off or deliberately misleading, DoD has a very grave problem on its hands,” says R. David Edelman, who served as special assistant to former President Barack Obama on cybersecurity and tech policy. “In the private sector, this is the sort of report that would put the CEO on death watch.”

DoD testers found significant vulnerabilities in the department’s weapon systems, some of which began with poor basic password security or lack of encryption. As previous hacks of government systems, like the breach at the Office of Personnel Management or the breach of the DoD’s unclassified email server, have taught us, poor basic security hygiene can be the downfall of otherwise complex systems.'''

text = text.replace('\n', ' ')
nlp = en_core_web_sm.load()
kg = KG()
kg.doc_dict = {1: nlp(text)}
kg.entity_detection()
pickle.dump(kg, open('kg.p', 'wb'))

# -*- coding: utf-8 -*-
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
import textacy
import re
from spacy.attrs import LEMMA, LIKE_NUM , IS_STOP
from spacy import displacy
from collections import Counter
nlp = spacy.load('en_coref_md')

def caps_abrev(caps, full):
    ## caps should be a token where caps stand for the capitalized words in full
    ## full should be a span
    caps = re.sub("[a-z]", "", caps.text)
    i = 0
    for l in full:
        c = l.text[0]
        if l.text.isupper() or not c.isupper():
            continue
        if i >= len(caps) or c != caps[i]:
            return False
        i += 1
    return i == len(caps)



def can_merge_span(span1, span2):
    # All strings mapped to integers, for easy export to numpy

    np_array1 = span1.to_array([LEMMA, LIKE_NUM, IS_STOP])
    np_array1 = np.apply_along_axis(lambda x:  x[0] if x[1] or not x[2] else -1 , 1,np_array1 )
    # print(np_array1)

    np_array2 = span2.to_array([LEMMA, LIKE_NUM, IS_STOP])
    np_array2 = np.apply_along_axis(lambda x: x[0] if x[1] or not x[2] else -1,
                                    1, np_array2)
    # print(np_array2)
    score = np.intersect1d(np_array1, np_array2).size / np.union1d(np_array1, np_array2).size
    if score > 0.8:
        print("Entity Merge: " + span1.text + " and "+ span2.text + " because score = "+ str(score))
        return True

    s1 = np.array(span1)[np_array1 != -1]
    s2 = np.array(span2)[np_array2 != -1]

    if s1.size > 1 and s2.size == 1:
        if(caps_abrev(s2[0], s1)):
            print("Entity Merge: " + span1.text + " and " + span2.text + " because " +
                  str(s2) + " stands for" + str(s1))

            return True

    return False


class Entity:
    '''The entity class is for storing named entities with the KG. These will
    eventually be used to create the nodes of the KG.'''
    def __init__(self, name, index, entity):
        #The plain text representation
        self.name = name
        # spacy entity object
        self.entity = entity
        #A set containing all plain text representations of the same entity
        self.aliases = {name}
        #The unique integer index value given by the KG to the entity
        self.index = index
        #set of appearences in the document corpus formatted as (doc_ix, token_ix_start, token_ix_end)
        self.doc_appearances = set()
        #Spacy entity label
        self.ent_class = entity.label



    def merge(self, other_ent):
        '''Updates the entity to contain the aliases and appearences of another entity
        object representing that same underlying entity.'''

        # print(sim)
        if can_merge_span(self.entity, other_ent.entity):
            print(self.name + " merge with " + other_ent.name )
            self.aliases = self.aliases.union(other_ent.aliases)
            self.doc_appearances.union(other_ent.doc_appearances)
            return True
        return False


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
        # {doc_ix : {token_ix: entity_id}}
        self.master_token_ix_to_entity = {}
        # {document id: spacy doc object}
        self.doc_dict = {}
        # The number of unique index values given out for entities
        self.keys = 0
        #set of all triple relationships in the form of (subj, vb, obj)
        self.triples = set()
        #
        self.relations = {}

    def add_new_entity(self, doc_ix, ent_span):
        '''Updates the KG data fields and creates new entity object.
        Return the id of the newly created entity.
        doc_ix: int - the document index
        ent_span: spacy.span - the span object associated with the new entity'''
        ent_id = self.keys
        new_ent = Entity(ent_span.text, ent_id, ent_span)
        new_ent.doc_appearances.add((doc_ix, ent_span.start, ent_span.end))
        self.entities[ent_id] = new_ent
        self.name_to_ix[ent_span.text] = ent_id
        self.ix_to_name[ent_id] = ent_span.text
        for i in range(ent_span.start, ent_span.end):
            self.master_token_ix_to_entity[doc_ix][i] = ent_id
        self.keys += 1
        return ent_id

    def entity_detection(self):
        '''Compile a list of entities from a collection of documents.
        doc_dict - {document index: spacy Doc object}'''
        for ix, doc in self.doc_dict.items():
            self.master_token_ix_to_entity[ix] = {}
            for ent in doc.ents: #For all entities in all documents
                if ent.text in self.name_to_ix:
                    #If ent already exists, add appearence
                    ent_id = self.name_to_ix[ent.text]
                    self.entities[ent_id].doc_appearances.add((ix, ent.start, ent.end))
                    for i in range(ent.start, ent.end):
                        self.master_token_ix_to_entity[ix][i] = ent_id
                else:
                    #Else create new entity and update KG data fields
                    self.add_new_entity(ix, ent)

    def update_entity_appearance_records(self, doc_ix, ent_id, cluster, multi=False):
        '''Updates entites doc_appearances field and knowledge graph
        master_token_ix_to_entity field to accound for coreferences.
        doc_ix : int - the document index of the cluster
        ent_id : int or list - the entity id(s) to be updated
        cluster : cluster object - contains references for updates
        multi : bool - true if ent_id is a list'''
        for mention in cluster.mentions:
            for i in range(mention.start, mention.end):
                self.master_token_ix_to_entity[doc_ix][i] = ent_id
                if multi:
                    for ent in ent_id:
                        self.entities[ent].doc_appearances.add((doc_ix, mention.start, mention.end))
                else:
                    self.entities[ent_id].doc_appearances.add((doc_ix, mention.start, mention.end))


    def coreference_detection(self):
        '''Updates entity and knowledge graph data to account for
        coreferences to entities (such as pronouns) in text.'''
        #For each document
        for ix, doc in self.doc_dict.items():
            clusters = doc._.coref_clusters
            #For each coreference cluster in the document
            for cluster in clusters:
                #Get the entity(s) associated with the cluster head
                head = cluster.main
                head_ents = []
                head_start, head_end = head.start, head.end
                for i in range(head.start, head.end):
                    ent_ref = self.master_token_ix_to_entity[ix].get(i, -1)
                    if ent_ref != -1:
                        head_ents.append(ent_ref)
                #If there is no assocated entity
                if len(head_ents) == 0:
                    #Create a new entity with appearances including the corefs
                    ent_id = self.add_new_entity(ix, ent_span)
                    self.update_entity_appearance_records(ix, ent_id, cluster)
                #If there are one or more associated entites, update appearance records
                elif len(head_ents) == 1:
                    head_ent = head_ents[0]
                    self.update_entity_appearance_records(ix, head_ent, cluster)
                else:
                    self.update_entity_appearance_records(ix, head_ents, cluster, True)


    def get_pos(doc, pos_name):
        "get list of pos_name entities from parsed document"
        pps = []
        for token in doc:
            if token.pos_ == pos_name:
                pp = ' '.join([tok.orth_ for tok in token.subtree])
                pps.append(token.subtree) #pp
        return pps

    def merge_entities(self):
        #TODO: update master_token_ix_to_entity
        result = {}
        for i in self.entities:
            e = kg.entities[i]
            merged = False
            for j in result:
                if result[j].merge(e):
                    merged = True
            if not merged:
                result[i] = e
        self.entities = result

    def triple_extraction(self):
        '''
        extracts triple relationships in text,
        stored as 3-tuples in self.triples

        each triple object is tuple of span objects
        '''
        #identify obvious subject-verb-object triples
        for idx, doc in self.doc_dict.items():
            text_ext = textacy.extract.subject_verb_object_triples(doc)

            for x in text_ext:
                self.triples.add(x)

        #add all subjects and objects to entities list if not present
        ents = set(list(map(lambda c: self.entities[c].name, self.entities.keys())))
        for tup in self.triples:
            if tup[0].text not in ents:
                self.entities[len(self.entities)] = Entity(tup[0].text, len(self.entities), tup[0])

            if tup[2].text not in ents:
                self.entities[len(self.entities)] = Entity(tup[2].text, len(self.entities), tup[2])

        #TODO: parse grammar trees for relationships
        #TODO: get all entities
        #TODO: account for prepositions 'ADP' & other special cases (which)


    def graph_construction(self):
        pass


text = '''The first step in solving any problem is admitting there is one. But a new report from the US Government Accountability Office finds that the Department of Defense remains in denial about cybersecurity threats to its weapons systems.

Specifically, the report concludes that almost all weapons that the DoD tested between 2012 and 2017 have “mission critical” cyber vulnerabilities. “Using relatively simple tools and techniques, testers were able to take control of systems and largely operate undetected, due in part to basic issues such as poor password management and unencrypted communications,” the report states. And yet, perhaps more alarmingly, the officials who oversee those systems appeared dismissive of the results.

The GAO released its report Tuesday, in response to a request from the Senate Armed Services Committee ahead of a planned $1.66 trillion in spending by the Defense Department to develop its current weapons systems. Subtitled "DoD Just Beginning to Grapple with Scale of Vulnerabilities," the report finds that the department "likely has an entire generation of systems that were designed and built without adequately considering cybersecurity." Neither Armed Services Committee chairman James Inhofe nor ranking member Jack Reed responded to requests for comment.

The GAO based its report on penetration tests the DoD itself undertook, as well as interviews with officials at various DoD offices. Its findings should be a wakeup call for the Defense Department, which the GAO describes as only now beginning to grapple with the importance of cybersecurity, and the scale of vulnerabilities in its weapons systems.

“I will say that the GAO can be prone to cyber hyperbole, but unless their sampling or methodology were way off or deliberately misleading, DoD has a very grave problem on its hands,” says R. David Edelman, who served as special assistant to former President Barack Obama on cybersecurity and tech policy. “In the private sector, this is the sort of report that would put the CEO on death watch.”

DoD testers found significant vulnerabilities in the department’s weapon systems, some of which began with poor basic password security or lack of encryption. As previous hacks of government systems, like the breach at the Office of Personnel Management or the breach of the DoD’s unclassified email server, have taught us, poor basic security hygiene can be the downfall of otherwise complex systems.'''

text = text.replace('\n', ' ')
kg = KG()
kg.doc_dict = {1: nlp(text)}

print("calling entity detection")
kg.entity_detection()
print("number of entities now: {}".format(len(kg.entities)))

print("calling coreference detection")
kg.coreference_detection()
print("number of entities now: {}".format(len(kg.entities)))

print("calling merge entities")
kg.merge_entities()
print("number of entities now: {}".format(len(kg.entities)))

print("calling triple extraction")
kg.triple_extraction()
print("number of entities now: {}".format(len(kg.entities)))


print("#######PRINTING ENTITIES#######")

for i in kg.entities:
    print(kg.entities[i].name)
    print(kg.entities[i].doc_appearances)


print("#######PRINTING TRIPLES#######")
for tup in kg.triples:
    #print(type(tup[0].text))
    print(tup)


#print(kg.entities.keys())



#pickle.dump(kg, open('kg.p', 'wb'))

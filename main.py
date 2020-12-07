# for import/natural language processing
!pip install spacy
import spacy
import pickle
import re
import pathlib
from pathlib import Path

#infomap/louvain
!pip install infomap
!pip install cdlib
!pip install wurlitzer 
import cdlib
from cdlib import algorithms, viz, evaluation
from cdlib.algorithms import eva
import infomap

#networkx
import networkx as nx

#functions/dependencies
def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.rstrip()
    return text.lower()

nlp = spacy.load("en_core_web_sm")

#import and clean .txt file
txt = Path('CISI_CRAN_200_test.txt').read_text()
txt = txt.replace('\n', ' ')
txt = text_cleaner(txt)
txt = txt.split('.i')

corpus = []
corpus.append(txt)

with open("dataset.txt", "wb") as fp:   # Pickling
    pickle.dump(corpus, fp)

with open('dataset.txt', 'rb') as fp:   # Unpickling
    corpus = pickle.load(fp)

#pulls subject indicators from documents and assigns each subject indicator a unique integer value
#entities is a dictionary where each document is a key and the values are subject indicators
unique_word = []
entities = {}
for idx, article in enumerate(corpus[0], 1):
    tokens = nlp(''.join(article))
    subject_list = []
    for ent in tokens.ents:
        if ent.label_ == 'PERSON':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'NORP':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'EVENT':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'WORK_OF_ART':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'LAW':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'LANGUAGE':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'GPE':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'ORG':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'LOC':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'FAC':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'PRODUCT':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        if ent.label_ == 'DATE':
            subject_list.append(ent.text)
            unique_word.append(ent.text)
        #if ent.label_ == 'CARDINAL':
            #subject_list.append(ent.text)
            #unique_word.append(ent.text)
    store = [unique_word.index(x) for x in subject_list]
    entities[idx] = store

#removes documents from entities that don't have subject indicators
filtered_entities = {k:v for k,v in entities.items() if v != []}
print(filtered_entities)

#creates graph
G = nx.Graph(filtered_entities)

print(G.nodes())
print(G.edges())

#remove outliers/self-loops
G.remove_edges_from(nx.selfloop_edges(G))
G = nx.k_core(G,k=2)

#Louvain/infomap algorithm and graph plot
#coms = algorithms.louvain(G)
coms = algorithms.infomap(G)
pos = nx.spring_layout(G)
viz.plot_community_graph(G, coms, figsize=(8, 8), node_size=200, plot_overlaps=False, plot_labels=True, cmap=None, top_k=None, min_size=None)
viz.plot_network_clusters(G, coms, position=None, figsize=(8, 8), node_size=200, plot_overlaps=False, plot_labels=False, cmap=None, top_k=None, min_size=None)

#converting this to an nx graph for calculations.
mod = evaluation.modularity_density(G,coms)
print(mod)

#calculating modularity
mod = evaluation.modularity_density(G,coms)
print(mod)

#calculating purity
#communities = eva(G, coms)
#pur = evaluation.purity(communities)
#print(pur)

#calculating avg embeddedness
ave = evaluation.avg_embeddedness(G,coms)
print(ave)

#calculating avg distance
scd = evaluation.avg_distance(G,coms)
print(scd)

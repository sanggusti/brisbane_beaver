'''
Author: Louis Owen (https://louisowen6.github.io/)
'''

import json
import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plagiarism_score_graph(score_dict,dict_path,plagiarism_thres=0.6):
    unique_labels = []
    for key_pairs in score_dict:
        for i in range(2):
            if key_pairs[i] not in unique_labels:
                unique_labels.append(key_pairs[i])
    
    plagiarism_dict = {k:v for k, v in score_dict.items() if v > plagiarism_thres}
    plagiarism_pairs = list(plagiarism_dict.keys())
    weights = list(plagiarism_dict.values())
    
    edges = []
    for pairs in plagiarism_pairs:
        edges.append((unique_labels.index(pairs[0]), unique_labels.index(pairs[1])))
        
    G=nx.Graph()
    G.add_nodes_from(range(len(unique_labels)))
    G.add_edges_from(edges)

    # some labels
    labels = {}
    for i in range(len(unique_labels)):
        labels[i] = unique_labels[i]
    
    d = json_graph.node_link_data(G)

    for i in range(len(d['nodes'])):
        d['nodes'][i]['id'] = labels[d['nodes'][i]['id']]

    chosen_links = []
    for i in range(len(d['links'])):
        if weights[i] > plagiarism_thres:
            chosen_links.append(
                                {
                                    'source': labels[d['links'][i]['source']],
                                    'target': labels[d['links'][i]['target']],
                                    'value': weights[i]
                                }
                                )
    
    d['links'] = chosen_links
    
    with open(dict_path,'w') as f_out:
        json.dump(d,f_out)
        
    return d

    
def color_mapper(lst):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.turbo)
    
    color_lst = []
    for v in lst:
        color_lst.append(mapper.to_rgba(v))
        
    return color_lst
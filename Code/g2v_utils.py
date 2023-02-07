#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importations
import networkx as nx
import matplotlib.pyplot as plt
import os 
import glob
import numpy as np
import umap
import seaborn as sns
import pandas as pd
from karateclub import EgoNetSplitter
from karateclub import Graph2Vec
from compress_pickle import load as cload
from compress_pickle import dump as cdump
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from g2v_utils import *


# In[1]:


# Utilities
def plt_style(size=(15,9),
              titleSize=55,
              labelsSize=50,):
    # Figure
    plt.figure(figsize = size)

    # Font sizes
    plt.rcParams['font.size'] = titleSize
    plt.rcParams['axes.labelsize'] = labelsSize
    plt.rcParams['axes.titlesize'] = titleSize
    plt.rcParams['xtick.labelsize'] = labelsSize
    plt.rcParams['ytick.labelsize'] = labelsSize
    plt.rcParams['legend.fontsize'] = labelsSize
    plt.rcParams['figure.titlesize'] = titleSize
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rcParams["axes.grid"] = False

    # axes
    ax = plt.subplot(111)                    
    #ax.spines["top"].set_visible(False)  
    #ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    
    return ax

# Check graph inside nxgraphs using index (idx) and attribute (e.g., supply)
def check_graph(idx, attribute=''):
    print(nx.info(nxgraphs[idx]))
    try:
        if attribute != '':
            print("AVG " + attribute + ":", np.mean([i[0] for i in nx.get_node_attributes(nxgraphs[idx], attribute).values()]))
            print("std " + attribute + ":", np.std([i[0] for i in nx.get_node_attributes(nxgraphs[idx], attribute).values()]))
            print("Max " + attribute + ":", np.max([i[0] for i in nx.get_node_attributes(nxgraphs[idx], attribute).values()]))
            print("Min " + attribute + ":", np.min([i[0] for i in nx.get_node_attributes(nxgraphs[idx], attribute).values()]))
        else:
            print("No attribute provided")
    except:
        print("Attribute", attribute, "not found")
    nx.draw(nxgraphs[idx], edge_color='k')
    
# Check graph inside nxgraphs using index (idx) and attribute (e.g., supply)
def check_graph_V2(idx, attribute=''):
    print(nx.info(nxgraphs[idx]))
    try:
        if attribute != '':
            print("AVG " + attribute + ":", np.mean([i for i in nx.get_node_attributes(nxgraphs[idx], attribute).values()]))
            print("std " + attribute + ":", np.std([i for i in nx.get_node_attributes(nxgraphs[idx], attribute).values()]))
            print("Max " + attribute + ":", np.max([i for i in nx.get_node_attributes(nxgraphs[idx], attribute).values()]))
            print("Min " + attribute + ":", np.min([i for i in nx.get_node_attributes(nxgraphs[idx], attribute).values()]))
        else:
            print("No attribute provided")
    except:
        print("Attribute", attribute, "not found")
    nx.draw(nxgraphs[idx], edge_color='k')
    
# Check graph wmap
def check_graph_wmap(h):
    # Info
    print(nx.info(h))
    
    # Gather position of the countries
    dpos = {}
    for n, a in h.nodes(data=True):
        try:
            dpos[n] = all_centroids[inv_nodes_map[n]]
        except:
            pass

    # Data
    print("AVG supply:", 
          np.round(np.mean([i for i in nx.get_node_attributes(h, 'supply').values()])),3)
    print("AVG demand:", 
          np.round(np.mean([i for i in nx.get_node_attributes(h, 'demand').values()])),3)
    print("AVG degree:", 
          np.round(np.mean([i for i in nx.get_node_attributes(h, 'degree').values()])),3)
    print("AVG in_degree:", 
          np.round(np.mean([i for i in nx.get_node_attributes(h, 'in_degree').values()])), 3)
    print("AVG out_degree:", 
          np.round(np.mean([i for i in nx.get_node_attributes(h, 'out_degree').values()])), 3)
    print("AVG betweenness:", 
          np.round(np.mean([i for i in nx.get_node_attributes(h, 'betweenness').values()])), 3)
    print("AVG DPV SC:", 
          np.round(np.mean([i for i in nx.get_node_attributes(h, 'dpv_sc').values()])), 3)
    print("AVG DPV DC:", 
          np.round(np.mean([i for i in nx.get_node_attributes(h, 'dpv_dc').values()])), 3)
        
    # Plot
    ax = plt_style(size=(20,20))
    world.boundary.plot(ax=ax, ec='k')
    nx.draw(h, 
            pos=dpos,
            node_size=100,
            edge_color='gray',
            with_labels=True,
            ax=ax)    

def nodesExtract(dfNetworks, verbose=False):
    '''
    input:
    output: list of all nodes list of all their centroids (lat,lon)
    '''
    # Loop over nodes and relabel
    all_nodes = []
    all_centroids = {}
    # verbose = False
    nxgraphs_raw = dfNetworks.network.tolist() 
    # Extract all nodes across graphs
    for g in nxgraphs_raw:
        for u, a in g.nodes(data=True):
            if u not in all_nodes:
                all_nodes.append(u)
                all_centroids[u] = [a['lon'], a['lat']]
            if verbose:
                print(u)
                print(a)
                print()
    return np.sort(all_nodes), all_centroids, nxgraphs_raw
    
def cleanGraphs(nxgraphs_raw,nodes_map,inv_nodes_map, verbose=False):
    nxgraphs =[]
    # Clean raw graphs
    for g in nxgraphs_raw:
        # Map all nodes
#         print(g)
        h = nx.relabel_nodes(g, nodes_map)
        print(h)
        # Delete node attributes
        for u in h:
            del h.nodes[u]['lat']
            del h.nodes[u]['lon']
            del h.nodes[u]['continent']
            del h.nodes[u]['region']
            del h.nodes[u]['dpv']
            del h.nodes[u]['dpv_s']
            del h.nodes[u]['dpv_d']
            h.nodes[u]['demand'] = np.round(h.nodes[u]['demand'], 3)
            h.nodes[u]['supply'] = np.round(h.nodes[u]['supply'], 3)
            h.nodes[u]['dpv_sc'] = np.round(h.nodes[u]['dpv_sc'], 3)
            h.nodes[u]['dpv_dc'] = np.round(h.nodes[u]['dpv_dc'], 3)
            h.nodes[u]['degree_cen'] = np.round(h.nodes[u]['degree_cen'], 5)
            h.nodes[u]['betweenness'] = np.round(h.nodes[u]['betweenness'], 5)
        # Delete edges attributes
        for _, _, d in h.edges(data=True):
            d.pop('attr_dict')
            d['weight'] = np.round(d['weight'], 3)
        # Add dummy nodes
        for i in nodes_map.values():
            if not i in h.nodes:
                if verbose:
                    print('Adding', i)
                h.add_node(i)
                h.nodes[i]['demand'] = 0.
                h.nodes[i]['supply'] = 0.
                h.nodes[i]['dpv_sc'] = 0.
                h.nodes[i]['dpv_dc'] = 0.
                h.nodes[i]['degree_cen'] = 0.
                h.nodes[i]['degree'] = 0
                h.nodes[i]['in_degree'] = 0
                h.nodes[i]['out_degree'] = 0
                h.nodes[i]['betweenness'] = 0.
        # Save processed graph
        nxgraphs.append(h)
#         print(nxgraphs)
        # Total graphs
        print("Total processed graphs:", len(nxgraphs))       
    return nxgraphs


# In[ ]:


# dat_with_clusters
def addxFeats(DF,dfNetworks,
              col_names= ['product','year','nodes','edges','avg_clustering_coef']):
    ## appends features to DF
    for name in col_names:
        if name == 'product':
            DF[name] =  dfNetworks.copy().loc[:, name]
        else:
            DF[name] =  dfNetworks.copy().loc[:,name].astype('int')

def group_plotter(DF, groupby= 'product', color = 'gray',agg=''):
    # plt.title('Product vs year scatter plot with clusters')
    f = plt.figure(figsize=(16,10))
    ## store tuples of (cluster_num, labels) in Labels lists
    for i in range(2, 8):
        f.add_subplot(2, 3, i-1)
        plt.bar(range(i),DF.groupby(f'Clusters_{i}').count()[groupby], color=color)
        plt.xlabel('Clusters')
        plt.title(f'{agg}Clusters_{i}')
    plt.show() 

# plt.figure()
# for n_clusters, labels in Labels:
def group_stats(Labels, DF, thresh1=0.5,
                thresh2= 0.33, agg=''):
    # ## unique products in each group 
    for n_clusters, labels in Labels:
        for i in range(n_clusters):
            print('-------  --------  --------')
            print(f"{agg}cluster_{n_clusters}")
            print('-------  --------  --------')
            print(f'Fraction of product networks in group_{i} cluster_{n_clusters}')
            # print(data_with_clusters[data_with_clusters.Clusters==0].products.unique())
            temp = DF[DF[f'Clusters_{n_clusters}']==i]
            temp = temp.groupby('product').count()[f'Clusters_{n_clusters}'].sort_values(ascending=False)/25
            display(temp[temp>thresh1])
            print(" ")
            print('-------  --------  --------')
            print(" ")
            print(f'unique products (filtered) in group_{i} cluster_{n_clusters}')
            print(list(temp[temp>thresh2].index)) 
            print(" ")


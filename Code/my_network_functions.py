
import itertools
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter


### Network Functions
def makeNetworkFromFiles(dataset):
    ''' 
    Function takes dataframe as input and generates weighted(trade imports) network
    '''
    G = nx.DiGraph()  #modified from lecture 12 
    # Add edges and edge attributes
    for i, elrow in dataset.iterrows():
        G.add_edge(elrow[0], elrow[1], weight =elrow[2],  attr_dict=elrow[3:].to_dict())
    return G

## Function to remove values from list
def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

## Analysis
def degree_dist(G):
    ''' Function takes as input a networkx directed graph, G and 
    return the regular and weighted in(out)degree of each node in 4 separate lists'''
    degrees = []
    out_weighted = []
    out_d= []
    in_weighted = []
    in_d= []

    ## Degree of network
    degrees=list(dict(G.degree()).values()) 
    ## Outdegree of network
    out_d=list(dict(G.out_degree()).values())
    ## Indegree of network
    in_d=list(dict(G.in_degree()).values())

    ## Eliminate nodes with indegree with ZEROs(no flows)
    in_d=remove_values_from_list(in_d,0)
    out_d=remove_values_from_list(out_d,0)

    ## create weighted out degree and in degree list
    out_weighted=list(dict(G.out_degree(weight='weight')).values())
    in_weighted=list(dict(G.in_degree(weight='weight')).values())
    ## Eliminate ZEROS
    out_weighted=remove_values_from_list(out_weighted,0)
    in_weighted=remove_values_from_list(in_weighted,0)
    return in_d, out_d, in_weighted, out_weighted

def deg_dist_plotter(in_d, out_d,name, year='global', n_bins = 20, weighted = False):
    ## create figure
    fig, ax = plt.subplots()
    fig.set_size_inches((9, 7))
    
    if weighted == False:
        ## log-log plot of the probability distribution of connectivity
        
        #n, bins = np.histogram(out_d, bins = range(min(out_d), max(out_d)+1, 2), normed="True") 
        out_logBins = np.logspace(np.log10(min(out_d)), np.log10(max(out_d)),num=n_bins)
        out_logBinDensity, out_binedges = np.histogram(out_d, bins=out_logBins, density=True)

        #n, bins = np.histogram(in_d, bins = range(min(in_d), max(in_d)+1, 2), normed="True") 
        in_logBins = np.logspace(np.log10(min(in_d)), np.log10(max(in_d)),num=n_bins)
        in_logBinDensity, in_binedges = np.histogram(in_d, bins=in_logBins, density=True)

        ax.loglog(out_logBins[:-1],out_logBinDensity,'o', markersize=10,label=r'$k_{in}$')
        ax.loglog(in_logBins[:-1],in_logBinDensity,'s', markersize=10,label=r'$k_{out}$')
        ax.legend(fontsize=30)


        ax.set_xlabel('$degree, k$',fontsize=40)
        ax.set_ylabel('$P(k)$',fontsize=40) 
        names ="images/{} - degree ".format(name)+str(year)+".png"
        plt.savefig(names)
        
    else:
        #n, bins = np.histogram(out_weighted, bins = range(min(out_weighted), max(out_weighted)+1, 2), normed="True") 
        out_logBins = np.logspace(np.log10(min(out_d)+0.001), np.log10(max(out_d)),num=n_bins)
        out_logBinDensity, out_binedges = np.histogram(out_d, bins=out_logBins, density=True)

        #n, bins = np.histogram(in_weighted, bins = range(min(in_weighted), max(in_weighted)+1, 2), normed="True") 
        in_logBins = np.logspace(np.log10(min(in_d)+0.001), np.log10(max(in_d)),num=n_bins)
        in_logBinDensity, in_binedges = np.histogram(in_d, bins=in_logBins, density=True)

        ax.loglog(out_logBins[:-1],out_logBinDensity,'o', markersize=10,label=r'$w_{in}$')
        ax.loglog(in_logBins[:-1],in_logBinDensity,'s', markersize=10,label=r'$w_{out}$')
        ax.legend(fontsize=30)

        ax.set_xlabel('weighted degree, $w$',fontsize=40)
        ax.set_ylabel('$P(w)$',fontsize=40) 
        names = "images/{} - weighted degree".format(name) +str(year)+".png"
        plt.savefig(names)
        
def plotter_2(G,name,year='global'):
    kk = []
    wij = []
    degrees = G.degree()
    for n in G.nodes(data=True):
        for e in G.edges(n[0],data=True):
            kk.append(degrees[e[0]]*degrees[e[1]])
            wij.append(G[e[0]][e[1]]['weight'])

    n_bins = 40
    kk_logBins = np.logspace(np.log10(min(kk)), np.log10(max(kk)),num=n_bins)
    counts, bins = np.histogram(kk, bins=kk_logBins);
    sums, bins = np.histogram(kk, bins=kk_logBins,weights=wij);
    avg_w = sums/counts; 

    fig, ax = plt.subplots()
    fig.set_size_inches((18, 7))
    ax.loglog(bins[:-1],avg_w,linewidth=0,color='r',marker='o',markersize=10)
    ax.set_xlabel('$k_ik_j$',fontsize=20)
    ax.set_ylabel('$<w_{i,j}>$',fontsize=20)
    names = "images/{} - weighted_directed_".format(name) +str(year)+".png"
    plt.savefig(names)
    
def plotter(x,y,name,name_x, name_y,color='k',marker='o',markersize=10):
    fig, ax = plt.subplots()
    fig.set_size_inches((18, 7))
    plt.plot(x,y,color=color,marker=marker,markersize=10)
    ax.set_xlabel(name_x,fontsize=20)
    ax.set_ylabel(name_y,fontsize=20)
    names = "images/ {} - ".format(name)+name_y+" versus " +name_x+".png"
    plt.savefig(names)
    
def stats(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ## average clustering coeff
    avg_C = nx.average_clustering(G)
    print('Clustering coefficient: ',avg_C )
    ## network density: ratio of actual edges in the network to all possible edges in the network.
    density = nx.density(G)
    print("Network density:", density)
    
    # check if your G is weakly connected
    print(nx.is_connected(G.to_undirected()))
    
    #Transitivity
    triadic_closure = nx.transitivity(G)
    print("Triadic closure:", triadic_closure)
    
    ## find degree dictionary and add to attributes
    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')
    ## sort by degree
    sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)
    top_degree = sorted_degree[:10]
    
    ## find weighted degree dictionary and add to attributes
    weighted_degree_dict = dict(G.degree(G.nodes(), weight = 'weight'))
    nx.set_node_attributes(G, weighted_degree_dict, 'weighted_degree')
    ## sort by weighted degree
    sorted_weighted_degree = sorted(weighted_degree_dict.items(), key=itemgetter(1), reverse=True)
    #First get the top 20 nodes by betweenness as a list
    top_weighted = sorted_weighted_degree[:10]
    mean_top_weighted = np.mean(sorted(dict(G.degree(G.nodes(), weight = 'weight')).values(), reverse=True)[:10])
    weighted_names = sorted(dict(G.degree(G.nodes(), weight = 'weight')).keys(), reverse=True)[:10]
    #Then find and print their degree
    for tw in top_weighted: # Loop through top_betweenness
        degree = degree_dict[tw[0]] # Use degree_dict to access a node's degree, see footnote 2
        print("Name:", tw[0], "| weighted degree :", tw[1], "| Degree:", degree)
        
    return [density,triadic_closure,nx.is_connected(G.to_undirected()),num_nodes, num_edges, avg_C,weighted_names, mean_top_weighted ]


## network and stats generator
def generator(data,name):
    Den = []
    Tran = []
    avg_flow = []
    Clus = []
    N = []
    E =[]
    names = []
    Gs = []
    ## add global network
    G = makeNetworkFromFiles(data)
    Gs.append(G)
    ## iterate to add each network of yearly flows
    years = data['Year'].unique()
    for year in years:
        dataset = data[data['Year'] == year]
        G = makeNetworkFromFiles(dataset)
        Gs.append(G)
        in_d, out_d, in_weighted, out_weighted =degree_dist(G)
        deg_dist_plotter(in_d, out_d,name, year,n_bins = 20)
        deg_dist_plotter(in_weighted, out_weighted,name,year,n_bins = 20, weighted = True)
        statistics = stats(G)
        Den.append(statistics[0])
        Tran.append(statistics[1])
        avg_flow.append(statistics[-1])
        N.append(statistics[3])
        E.append(statistics[4])
        Clus.append(statistics[5])
        names.extend(statistics[6])
#     plt.hist(names)
    print(set(names))
    plotter(years, N,name,  'Years','Nodes')
    plotter(years, E, name, 'Years','edges',color='purple')
    plotter(years, Clus,name, 'Years','clustering coeff',color='r')
    plotter(years, Den, name,'Years','Density',color='b')
    plotter(years, Tran,name, 'Years','Triadic_closure',color='g')
    plotter(years, avg_flow,name,'Years','mean import flows',color='orange') 
    return Gs
    


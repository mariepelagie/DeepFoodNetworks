#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import pickle
import lzma
## import my network functions from my python script
import my_network_functions as mx


# In[3]:


## impart countries, continents and regions data
path = 'Data/'
filename = path+'countryContinent.csv'
countries = pd.read_csv(filename,encoding='gbk')
countries = countries[['country', 'continent', 'sub_region','code_2']]
countries.head()


# In[4]:


## impart countries, continents and regions data
path = 'Data/'
filename = path+'lat_lon.csv'
coord = pd.read_csv(filename)

countries = pd.merge(countries, coord[['code', 'latitude', 
                           'longitude']] , how='left', left_on='code_2', right_on = 'code').drop(['code',
                                                            'code_2'], axis =1)


# ## Data Cleaning

# In[5]:


## Function fod cleaning data imported 
def data_cleaning(filename, countries,
                  cols = ['ReporterName','PartnerName','Quantity','TradeValue in 1000 USD',
               'Year','QtyUnit','Reporter Income Group','Partner Income Group']):   
     ## read data
    data = pd.read_csv(filename,encoding='gbk')
    ## Select Imports  and qty unit = kg
    data = data[(data['TradeFlowName'] == 'Import')&(data['QtyUnit'] == 'Kg')]
    ## remove non-countries
    data = data[~data['ReporterName'].isin(['European Union','All countries  All --- All  '])]
    data = data[~data['PartnerName'].isin(['Unspecified','Special Categories','Free Zones','European Union'])]

    ## select desired columns
    data = data[cols]

    ## add reporting country continent and subregion
    data_1 = pd.merge(data,countries , how='left', left_on='ReporterName', right_on = 'country')
    # data_1
    data_1.rename(columns= {'country': 'country_reporting', 'continent':'continent_reporting'
                           , 'sub_region' :'sub_region_reporting'}, inplace = True)
    ## add partner country continent and subregion
    data_1 = pd.merge(data_1,countries , how='left', left_on='PartnerName', right_on = 'country')
    # data_1
    data_1.rename(columns= {'country': 'country_partner', 'continent':'continent_partner'
                           , 'sub_region' :'sub_region_partner'}, inplace = True)

    ## drop extra columns
    data_1 = data_1.drop(['country_partner','country_reporting','QtyUnit'], axis =1 )

    ## sord by date and drop missing values
    dat_ = data_1.sort_values(by='Year', ascending=True,na_position='first').dropna()
    return dat_

    


# In[6]:


# names = ['corn','dairy','poultry','rice']
DF_list = {}
# names =['corn','beans','banana','chocolate', 'cocoa','dairy',
#         'fish', 'mango','orange','oats','rice','poultry']
names = ['corn','beans','banana','chocolate', 'cocoa','dairy',
               'barley', 'mango','orange','oats','rice','poultry',
            'apple','strawberry','yeast','coffee','sheep','lettuce','goat',
            'butter','honey','garlic','peas', 'chickpeas','brusselssprouts',
            'asparagus', 'potatoes','sweetPotatoes','onions','almonds','walnuts',
            'hazelnut','coconuts','chestnuts','pistachios','dates','figs','avocados',
            'lemons','grapefruit','papaws','cherries','kiwifruit','prunes','vanilla',
            'cloves','nutmeg','ginger','rye','millet','crab','lobster','mushrooms']
file_paths = [path+ name+'_blk_data.csv' for name in names]
i=0
for filename in file_paths:
    dat_ = data_cleaning(filename, countries)
    ## summary statistics of numeric columns
    print(filename, dat_.describe())

#     ## summary statistics of non-numeric columns
#     print(filename, dat_[['ReporterName', 'PartnerName', 
#             'continent_reporting', 'sub_region_reporting',
#            'continent_partner', 'sub_region_partner','Reporter Income Group','Partner Income Group']].describe())

#     print(f'number of years of data available for {filename}: ' , dat_['Year'].nunique())
#     print('min year: ', min(dat_['Year'].unique()),', max year: ', max(dat_['Year'].unique()))

    ## save DataFrames in list
    DF_list[names[i]] = dat_
    i+=1


# ## Convert DataFrame to Networks

# In[7]:


G_list =[]
#create directed network
network_list =[]
network_dict ={}
for ii in range(len(DF_list)):
    years = [1996+i for i in range(25)]
    for year in years:
        DF = DF_list[names[ii]][DF_list[names[ii]].Year ==year]
#         network_list.append(mx.makeNetworkFromFiles(DF))
        network_dict[(names[ii],year)] = mx.makeNetworkFromFiles(DF)


# In[8]:


# # nx.get_edge_attributes(G, 'weight')
# for e in G.edges(data=True):
#     print(e)
network_dict


# ### Node attributes:
# supply (positive number), demand (negative number), degree (out, in, and total), degree centrality, betweenness centrality, DPV, 
# 

# In[9]:


## calculate demand and supply of nodes
## Calculate  Demand  by countries
def get_demand_supply(data, year, verbose = False):
    dat_ = data[data.Year == year]
    demand = dat_.groupby(['ReporterName', 'continent_reporting'], 
                          as_index= False)['Quantity'].sum()
    # rename columns 
    demand = demand.rename(columns={'ReporterName': 'Country',
                                    'Quantity':'Demand', 'continent_reporting':'Region' })
    if verbose:
        display(demand.head(2))

    ## Calculate Supply by countries 
    supply = dat_.groupby(['PartnerName','continent_partner'], as_index= False)['Quantity'].sum()
    supply = supply.rename(columns={'PartnerName': 'Country',
                                    'Quantity':'Supply', 'continent_reporting':'Region' })
    # set supply to negative values
    supply['Supply'] = -1*supply['Supply']
    supply = supply.drop('continent_partner', axis=1)
    if verbose:
        display(supply.head(2))
    ## merge demand and supply
    dem_sup = pd.merge(demand[['Country','Demand']],supply,
                how="outer",on=['Country']).fillna(0)
    
    ## add lat_lon 
    dem_sup = pd.merge(dem_sup ,countries,
                how="inner",left_on=['Country'], right_on=['country']).drop(['country'], axis=1)
    return dem_sup


# In[10]:


# dat_ = DF_list[0]
dem_sup = get_demand_supply(DF_list[names[0]], 1997)
dem_sup


# In[11]:


## Tests if demand and supply node length the same
a = set(DF_list[names[0]][DF_list[names[0]].Year ==1997]['ReporterName']) 
b = set(DF_list[names[0]][DF_list[names[0]].Year ==1997]['PartnerName'])
len(a|b)


# In[12]:


def set_node_att(G, dem_sup, Verbose=False):
    
    ## set demand nodeattribute
    temp = dem_sup[['Country','Demand']]
    aa = temp.set_index('Country').to_dict('dict')
    nx.set_node_attributes(G, aa['Demand'], "demand")
    if Verbose:
        print('Number of demand nodes:', len(aa['Demand']))

    ## set supply node attribute
    temp = dem_sup[['Country','Supply']]
    ab = temp.set_index('Country').to_dict('dict')
    nx.set_node_attributes(G, ab['Supply'], "supply")
    if Verbose:
        print('Number of supply nodes:', len(ab['Supply']))
        
    ## set Latitude node attribute
    temp = dem_sup[['Country','latitude']]
    ac = temp.set_index('Country').to_dict('dict')
    nx.set_node_attributes(G, ac['latitude'], "lat")
    
    ## set Longitude node attribute
    temp = dem_sup[['Country','longitude']]
    ad = temp.set_index('Country').to_dict('dict')
    nx.set_node_attributes(G, ad['longitude'], "lon")
    
    ## set continent node attribute
    temp = dem_sup[['Country','continent']]
    ae = temp.set_index('Country').to_dict('dict')
    nx.set_node_attributes(G, ae['continent'], "continent")
   
    ## set region node attribute
    temp = dem_sup[['Country','sub_region']]
    af = temp.set_index('Country').to_dict('dict')
    nx.set_node_attributes(G, af['sub_region'], "region")
        
    
    ## set betweeness centrality node attribute
    bb = nx.betweenness_centrality(G)
    if Verbose:
        print('Check if betweeness is an instance of attribute dictionary:',isinstance(bb, dict))
    nx.set_node_attributes(G, bb, "betweenness")
    
    ## set degree centrality node attribute
    dc = nx.degree_centrality(G)
    if Verbose:
        print('Check if degree centrality is an instance of attribute dictionary:',isinstance(dc, dict))
    nx.set_node_attributes(G, dc, "degree_cen")
    
    ## degree 
    deg = dict(G.degree)
    if Verbose:
        print('Check if degree is an instance of attribute dictionary:',isinstance(deg, dict))
    nx.set_node_attributes(G, deg, "degree")

    ## in_degree 
    in_deg = dict(G.in_degree)
    if Verbose:
        print('Check if in_degree is an instance of attribute dictionary:',isinstance(in_deg, dict))
    nx.set_node_attributes(G, in_deg, "in_degree")

    ## out_degree 
    out_deg = dict(G.out_degree)
    if Verbose:
        print('Check if out_degree is an instance of attribute dictionary:',isinstance(out_deg, dict))
    nx.set_node_attributes(G, out_deg, "out_degree")
              
   
    return G


# In[13]:


## Test Attributes Function
G = set_node_att(network_dict[('corn', 1997)],
                 get_demand_supply(DF_list[names[0]], 1997, verbose = False),
                 Verbose = True)


# In[14]:


G.nodes['Macao']


# In[15]:


# Calculate dpv
def dpv_calculation(g,                     # Graph NX
                    epsilon = 0.5,         # Flow threshold
                    threshold_var='flow',  # Var used as threshold for DPV
                    var1='supply',         # DPV Var1
                    var2='demand',         # DPV Var2
                    verbose=True,):        # Verbosity bool
    
    # Initialize dpv fields
    for n in g.nodes():
        g.nodes[n]['dpv'] = 0
        g.nodes[n]['dpv_d'] = 0
        g.nodes[n]['dpv_s'] = 0
        g.nodes[n]['dpv_dc'] = 0
        g.nodes[n]['dpv_sc'] = 0
    
    # Main calculation loop
    for i in g.nodes():
        if verbose:
            print("Node:", i)
        Tree = nx.subgraph(g, {i} | nx.descendants(g, i))

        # Modfied degree
        d = Tree.degree(i)
        n = 0
        novalid_nodes = set([])
        for e in Tree.edges():
            if g[e[0]][e[1]][threshold_var] <= epsilon and e[0] == i:
                if verbose:
                    print("\tEdge:", e)
                n += 1
                novalid_nodes.add(e[1])
        delta_degree = d - n
        print("\tdelta degree:", delta_degree)


        # DPV
        g.nodes[i]['dpv_s'] = (sum([g.nodes[j][var1] for j in Tree.nodes if j != i]) + g.nodes[i][var1]) *\
                              delta_degree
        print("\tdpv_s:", g.nodes[i]['dpv_s'])

        g.nodes[i]['dpv_d'] = (sum([g.nodes[j][var2] for j in Tree.nodes if j != i]) + g.nodes[i][var2]) *\
                              delta_degree
        print("\tdpv_d:", g.nodes[i]['dpv_d'])

        # Corrected DPV
        g.nodes[i]['dpv_sc'] = (sum([g.nodes[j][var1] for j in Tree.nodes if j != i and j not in novalid_nodes]) +\
                                g.nodes[i][var1]) * delta_degree
        print("\tdpv_sc:", g.nodes[i]['dpv_sc'])

        g.nodes[i]['dpv_dc'] = (sum([g.nodes[j][var2] for j in Tree.nodes if j != i and j not in novalid_nodes]) +\
                                g.nodes[i][var2]) * delta_degree
        print("\tdpv_dc:", g.nodes[i]['dpv_dc'])
        
    # Return graph
    return g


# In[16]:


def add_dpv_att(G, verbose = True):
    # Initialize DPV attributes
    for n in G.nodes():
        G.nodes[n]['dpv'] = 0
        G.nodes[n]['dpv_d'] = 0
        G.nodes[n]['dpv_s'] = 0
        G.nodes[n]['dpv_dc'] = 0
        G.nodes[n]['dpv_sc'] = 0
    # Calculate DPV
    G = dpv_calculation(G,                     # Graph NX
                        epsilon = 0.5,         # Flow threshold
                        threshold_var='weight',  # Var used as threshold for DPV
                        var1='supply',         # DPV Var1
                        var2='demand',         # DPV Var2
                        verbose= verbose)         


# In[17]:


add_dpv_att(G,verbose = True)
G.nodes['Macao']


# 
# ## Create Outputs 

# In[18]:


#update networks
for ix in network_dict:
    name, year = ix
    G = set_node_att(network_dict[(name, year)],
                 get_demand_supply(DF_list[name], year, verbose = False),
                 Verbose = False)
    add_dpv_att(G,verbose = False)
    network_dict[ix] = G
    
    
    


# In[20]:


# Data Frame with Summary
Summary = pd.DataFrame(columns =['product','year','network','network type', 'nodes','edges', 
                                 'avg_in_deg','avg_out_deg'  ])

for ix in network_dict:
    name, year = ix
    G = network_dict[ix]
    Summary = Summary.append({'product': name, 'year': year, 'network': G, 
                              'network type': nx.info(G).split('\\n')[0].split(' ')[0],
                              'nodes':len(G.nodes), 'edges':len(G.edges), 
                             'avg_in_deg': np.mean(list(dict(G.in_degree).values())) ,
                             'avg_out_deg': np.mean(list(dict(G.out_degree).values()))}, 
                             ignore_index=True)
    
Summary


# In[21]:


## Output networks
# !pip install compress-pickle
from compress_pickle import dump as cdump
# networks = lzma.compress(G_list)
cdump(network_dict, 'final_lzmas/networks.lzma', compression='lzma')


# In[22]:


#nx.info(G)
# Data Frame with Summary
DF_tot = pd.DataFrame(columns =['country','product','year', 'demand','supply', 'lat',
                                'lon', 'continent','region','degree_cen',
                               'betweenness', 'degree','in_degree','out_degree',
                                'dpv','dpv_d','dpv_s','dpv_dc','dpv_sc'])

for ix in network_dict:
    name, year = ix
    G = network_dict[ix]
    for country in G.nodes:
        
        DF_tot = DF_tot.append({'country': country, 'product': name, 'year': year,  
                              'demand':G.nodes[country]['demand'], 'supply':G.nodes[country]['supply'], 
                                'lat':G.nodes[country]['lat'], 'lon':G.nodes[country]['lon'],
                                'continent':G.nodes[country]['continent'], 'region':G.nodes[country]['region'],
                              'degree_cen':G.nodes[country]['degree_cen'],'betweenness': G.nodes[country]['betweenness'],
                              'degree':G.nodes[country]['degree'],  'out_degree':G.nodes[country]['out_degree'],
                              'in_degree':G.nodes[country]['in_degree'],  'dpv':G.nodes[country]['dpv'],
                               'dpv_d':G.nodes[country]['dpv_d'],  'dpv_s':G.nodes[country]['dpv_s'],  
                                'dpv_dc':G.nodes[country]['dpv_dc'],
                                'dpv_sc':G.nodes[country]['dpv_sc'],
                               }, 
                             ignore_index=True)
    
DF_tot


# In[ ]:


import geopandas as gpd
gdf_tot = gpd.GeoDataFrame(
    DF_tot, geometry=gpd.points_from_xy(DF_tot.lon, DF_tot.lat))
gdf_tot


# In[ ]:


cdump(gdf_tot, 'final_lzmas/DF_total.lzma', compression='lzma')
cdump(Summary, 'final_lzmas/DF_networks.lzma', compression='lzma')


# ## Create outputs

# In[ ]:


# print(nx.info(G))
Summary = pd.DataFrame(columns =['product','network', 'nodes','edges', 
                                 'avg_in_deg','avg_out_deg'  ])
for i in range(len(G_list)):
    Summary = Summary.append({'product': names[i],'network': G_list[i],
                              'nodes':len(G.nodes), 'edges':len(G.edges), 
                             'avg_in_deg': np.mean(list(dict(G.in_degree).values())) ,
                             'avg_out_deg': np.mean(list(dict(G.out_degree).values()))}, 
                             ignore_index=True)
Summary


# In[ ]:


## Output networks
# !pip install compress-pickle
from compress_pickle import dump as cdump
# networks = lzma.compress(G_list)
cdump(G, 'networks.lzma', compression='lzma')


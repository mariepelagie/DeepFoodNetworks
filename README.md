# DeepFoodNetworks
The goal of this project is to develop a pipeline for categorizing supply chains(food) using structure and  risk levels and network risk prediction.  First we develop a framework to systematically group different food product supply chains by identifying their structural similarities and differences and vulnerability. Using the vector embeddings obtained from the graph neural network in out clustering pipeline, we train a supervised machine learning model to predict risk levels and identify supply chain networks with extreme risk levels.

Data: Obtained from a World Bank database consists yearly trade data between countries  for over $50$ different products for a period of $25$ years.
1. pipeline_test_V4_Jan_2022.ipynb
    This notebook cleans data for each product under the list (names) and converts these tables to networks and   
    later outputs csv files.
    inputs: countryContinent.csv
    output:
    a. Summary as 'DF_networks.lzma': data frame with columns =['product','year','network','network type', 
       'nodes','edges', 
                                 'avg_in_deg','avg_out_deg'  ]
    b. network_dict as ('networks.lzma'): dictionary of productXyear network
    c. 'DF_with_norm_dpv.lzma'
    d. gdf_tot as 'DF_total.lzma': geodataframe consisting of summary table plus geolocation of each countries
    helper module: my_network_functions
**Clustering**
2. Graph2Vec_cleaned.ipynb
    This notebook does the following:
    - Convert each example (product graph) into a 4 dimensional embedding using Graph2Vec embedder
    - Clustering:  use 2 clustering algorithms ( Spectral and Agglomerative) to group the different product
      networks. We carry out the following analysis:
        - investigate the right cluster size using silhouette profiles for spectral clustering and dendrogram for 
          agglomerative clustering.
        - Explore consistent cluster within ( changing random seed) and between selected algorithms
        - Further analysis
            - characteristics of clusters of size = 4 and 5
            - persistent groupings 
  inputs: 'DF_with_norm_dpv.lzma'
  outputs: No files create
  helper module: g2v_utils.py

**Predictions**
2. Graph2Vec_predictions.ipynb
   TBA
   inputs: 'DF_with_norm_dpv.lzma'
   outputs: 
   

# Recommendation System

* Tools Used:
   * Scikit-Learn
   * Pandas, Numpy, Matplotlib
   
* Dataset: https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding

* Approach:
    1. Grant-Writer and Recipient Data is clustered using K-Means. Number of clusters is chosen such that the number of points in each cluster is between 1000 and 10,000.
    2. The cosine similarities between centroids of each recipient cluster and lender cluster is stored in a Cluster Similarity Table. 
    3. A lender/recipient's past data is given to the classifier (kmeans.predict) to produce its cluster. The most similar recipient/lender cluster is picked from the Cluster Similarity Table.
    4. The points(vectors) in this cluster with the highest cosine similarities are the recommendations given to a user.
   

# Recommendation System

* Tools Used:
   * Scikit-Learn
   * Pandas, Numpy, Matplotlib
   
* Dataset: https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding

* Approach:
    1. Grant-Writer and Recipient Data is clustered using K-Means. Number of clusters is chosen such that the number of points in each cluster is between 1000 and 10,000.
    2. The distances between centroids of each recipient cluster and writer cluster is stored in a Cluster Distance Table. 
    3. Two Random Forest Classifiers are trained, one each on writer and recipient clustered data.
    4. A writer/recipient's past data is given to the classifier to produce its cluster. The closest recipient/writer cluster is picked from the Cluster Distance Table.
    5. The closest points in this cluster are the recommendations given to a user.
   

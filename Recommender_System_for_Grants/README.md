# Recommendation System

* Tools Used:
   * Scikit-Learn
   * Pandas, Numpy, Matplotlib
   
* Dataset: https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding

* Approach:
    1. Grant-Writer and Recipient Data is clustered using K-Means. Number of clusters is chosen such that the number of points in each cluster is between 1000 and 10,000
    2. Two Random Forest Classifiers are trained, one each on writer and recipient clustered data.
   

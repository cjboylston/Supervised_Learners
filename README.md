#Supervised Learning Comparison
Written by Christian Boylston (cboylston3)

System Requirements:
    
    - python 3.7
    - scikit-learn
    - matplotlib
    - pandas
    - numpy

Files:
    
    - adult.data.csv : A data set containing information from the 1994 US Census that enables a classification problem
    to label an individual as making greating than $50,000/year or not
    (Source: https://archive.ics.uci.edu/ml/datasets/adult).

    - phish.data.csv  : A data set containing aggregated information about phishing emails to classify them as legitimate
    or phishy (Source: https://archive.ics.uci.edu/ml/datasets/Website+Phishing)

    - SupervisedML.py : The code of the project that implements the Decision Tree, Neural Network, Boosted Decision Tree,
    K-Nearest Neighbors, and Support Vector Machine Algorithms.

    - cboylston3-analysis.pdf : an analysis of the results of the five different supervised learning algorithms and their
    performances on the datasets.

 Directions:
    
    1) Ensure that adult.data.csv and phish.data.csv are in the same directory as SupervisedML.py when SupervisedML.py is
    run.

    2) SupervisedML.py can be run by simply putting the command "python SupervisedML.py" in the terminal when in
    SupervisedML.py's working directory.

    3) SupervisedML.py will first create an 80/20 test train split over each dataset. It will then run the split data
    on all of the algorithms for the adult.data.csv data and display the confusion matrix and classification report
    for each of the algorithms. Then it will repeat this same process for phish.data.csv data. Then program will create
    the learning curves (which it will output) for each of the algorithms on the adult.data.csv data. It will repeat this
    process on the phish.data.csv data and then the program terminates. 

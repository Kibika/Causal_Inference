# causal_inference
This project focuses on implementing techniques used to perform causal inference to the Wisconsin cancer dataset.

The project begins with exploring the observed data, to derive the relationships that can seen from the data collected. 
The correlation matrix is plotted to identify the bivariate relationship between the variables.

Feature selection is done to reduce the dimensionality of the graph using the "feature_selection" script. The random forest classification model is used to extract the most important features.

The relationships under causal inference are also explored by using python libraries and knowledge of the relationship between radius, perimeter and area to plot directed acrylic graphs(DAGs).

The variables that directly cause the outcome variable as seen from the causal graph are retained in the dataset for training and
prediction. The script used for constructing the graph is the "causal_graph" script. The stability of the relationships between the independent variable in the graph is checked using Jaccard's similarity index.

The resulting graph from this analysis is shown below:
![causal7](https://user-images.githubusercontent.com/12167288/131225158-40a3f2cd-8293-4c29-a74d-f03f42ab0126.png)

The Bayesian network, fitted using the "training" script. The network train on discretized data, the data is discretized using the "discretizing" script. The network learns the condition probabilities of the nodes of the graph and uses this for prediction.
The Bayesian network achieves the metrics below.

Recall: 0.87

Accuracy: 0.95 

Precision: 0.97 

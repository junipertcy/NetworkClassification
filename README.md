# NetworkClassification

This repository contains the code for the M.S. thesis research titled
"Structure of Complex Networks across Domains" by Kansuke Ikehara (@kansuke231).
The thesis was enriched to [arXiv](https://arxiv.org/abs/1710.11304) with the titile, "
Characterizing the structural diversity of complex networks across domains".

## Data
First things first, we need data in order to analyze something. Code in this repo
 assumes the data, namely features of networks, is stored in a csv file named
 `features.csv` which looks like this:

```
.gmlFile,NetworkType,SubType,NumberOfNodes,NumberOfEdges,MeanDegree,MeanGeodesicDistance,...
some.gml,Social,Offline Social,62,152,4.9032,2.9455,...
another.gml,Social,Offline Social,62,152,4.9032,2.9455,...
```

The `.gmlFile`, `NetworkType`, and `SubType` are necessary fields,
while `NumberOfNodes`, `NumberOfEdges`, `MeanDegree`, etc. are features associated with the data.

## Analyses
The two main tasks in the research are accomplished by the two scripts, `one_by_many.py` and `multi_run.py`.

1. `one_by_many.py`: finding which features make a specific category of networks (i.e. online social networks) "stand out"
among others;
2. `multi_run.py`: finding out which pairs of network categories are often misclassified
by a machine learning classifier based solely on the structural features and their implication in the process,
constraint and growing mechanism of those networks (if two distinct categories of networks are misclassified often, are their similar in some sense?).

Lastly, if one wants to see how the data points in a 2D or 3D feature space, `plot.py` contains some code for 2D and 3D visualization.

## Example Usage
To discriminate features using an individual feature approach, we select a set of seed and non-seed nodes as representative classes for investigation (i.e. input csv file `dataset/nodes.csv`).

Second, for each representative class, we proceed to run binary classification (seed v.s. non-seed) `iter` times using
random forest, in which the seed nodes of interest correspond to the positive and the other correspond to negative.

A set of features for this task includes: `degree`, `betweenness`, `closeness`, `eigencentrality`, `coreness`, `layerness`, `pagerank`, `sum_friends_friends`, and `transitivity` (total of 9 features).

In each run: 
1. We split the data set into training and test sets with the ratio of `7:3` while preserving the ratio of class distribution.
2. Compute AUC.
3. Record the ranking of feature importance (sorted according to the Gini impurity decrease in the training phase of the RF classifier)

This is accomplished by the following script:
```commandline
python one_by_many.py --csv dataset/nodes.csv -f degree -f betweenness -f closeness -f eigencentrality -f coreness -f layerness -f pagerank -f sum_friends_friends -f transitivity --iter 100
```
The script outputs:
1. Average AUC score over the `iter` runs.
2. Aggreated rankings of feature importance.
3. A 2D plot for feature discrimination. The _x_-axis corresponds to the most important feature and the _y_-axis the second most important.

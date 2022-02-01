from __future__ import division
import numpy as np



#create list of true labels for generating classifier outputs
#nr_classes: number of possible class labels
#nr_examples_per_class: number of examples we want to generate of each class
#the length of the resulting list is nr_classes x nr_examples_per_class (1,...,1,2,...,2,3,...)
def create_true_labels(nr_classes, nr_examples_per_class):
    labels = []
    for i in range(nr_classes):
        cur_label = i + 1
        for j in range(nr_examples_per_class):
            labels.append(cur_label)
    labels = np.array(labels)
    return labels


#fuse according to Independent Opinion Pool
#dists: array of categorical distributions to fuse
#returns the fused categorical distribution
def iop(dists):
    prod = 1
    for d in dists:
        prod *= d
    return prod / np.sum(prod)
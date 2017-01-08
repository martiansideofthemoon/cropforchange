#!/usr/bin/env python

import numpy as np
import time
import itertools

# constants
num_crops = 5
num_plots = 16 #4x4

#number of training points
N = 10

plot_distribution = np.genfromtxt("distribution.csv",delimiter=',',autostrip=True)

# generating input data of crops
np.random.seed(int(time.time()))
def crop_numbers(num_crops):
    tmp = np.random.rand(num_crops)
    tmp = tmp/np.sum(tmp)
    return map(lambda x:int(x), (num_plots+1)*tmp)

def greedy_output(dist, perm, cropi):
    distro = np.copy(dist)
    crops = cropi[:]
    j_dist = [None for i in range(num_plots)]
    for crop in perm:
        while crops[crop]>0:
            max_arg = np.nanargmax(distro,axis=0)[crop:crop+1][0]
            if j_dist[max_arg] is None:
                j_dist[max_arg] = crop
                distro[max_arg] = np.nan
                crops[crop] = crops[crop]-1
            else:
                print "Houston! We got a problem."
    metric = 0
    for j in range(len(j_dist)):
        if j_dist[j] is not None:
            metric += dist[j, j_dist[j]]
    return metric, j_dist

max_metric = 0
max_dist = None

open("data.csv", "w+").close()
for iterations in range(N):
    i_crop = crop_numbers(num_crops)
    for permutation in itertools.permutations(range(num_crops)):
        j_metric, j_dist = greedy_output(plot_distribution, permutation, i_crop)
        if j_metric > max_metric:
            max_metric = j_metric
            max_dist = j_dist

    with open("data.csv", "a+") as f:
        for crop in i_crop:
            f.write(str(crop) + ",")
        for dist in max_dist:
            f.write(str(dist) + ",")
        f.write("\n")

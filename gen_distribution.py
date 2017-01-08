import argparse
import os
import random
import time

random.seed(time.time())

parser = argparse.ArgumentParser()
parser.add_argument('--num_crops', type=int, default=5,
                   help='Number of crops')
parser.add_argument('--num_plots', type=int, default=16,
                   help='Number of crops')

args = parser.parse_args()

data = []
for i in range(args.num_plots):
    distro = []
    for j in range(args.num_crops):
        distro.append(random.uniform(0,1))
    sum_val = sum(distro)
    distro = [x/sum_val for x in distro]
    data.append(distro)

output = ""
for distro in data:
    row = ""
    for k in distro:
        row += str(k) + ","
    output += row[:-1] + "\n"

with open("distribution.csv", "w") as f:
    f.write(output)
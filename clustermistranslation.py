#!/usr/bin/env python3

import subprocess

def cluster_submission(start=0, stop=7469, directory="../results/gb1/", prnum=500):
    call0 = "qsub -t {}-{} -cwd -e ../error/ -o ../output/ ".format(start, stop)
    call1 = "cluster_fitness.py {} {}".format(prnum, directory)
    subprocess.call(call0+call1, shell=True)

if __name__ == "__main__":
    with open("../data/gb1seqs.txt", "r") as f:
        gb1 = []
        for line in f.readlines():
            gb1.append(line.strip())
    cluster_submission(0, len(gb1))

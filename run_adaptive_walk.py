#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import multiprocessing as mp
from adaptive_walk import main

landscape = "pard3e2_1.db"
start_geno = "pard3e2_adaptive_walk_slct_low.txt"
results_dir = "../results/adaptive_walk/pard3e2/proteins1_popsize1e4/"

def run_simulation(ID):
    main(ID, int(1e4), landscape, start_geno, results_dir)
    return 1

if __name__ == "__main__":
    # Get all the sequences in
    slct = np.loadtxt("../data/"+start_geno, dtype=str)
    # Which sequences are already done?
    flist = os.listdir(results_dir)
    done = set([f.split("_")[0] for f in flist])
    # Get their index
    index = [i for i,seq in enumerate(slct) if not seq in done]
    # Start simulating
    pool = mp.Pool(processes=mp.cpu_count()-1)
    workers = [pool.apply_async(run_simulation, args=(id,)) for id in index]
    done = np.zeros((len(index),))
    i = 0
    for p in workers:
        done[i] = p.get()
        i += 1
    if not sum(done) == len(index):
        raise RuntimeError("Done has only {} entries".format(sum(done)))

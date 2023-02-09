#!/usr/bin/env python3

"""Infer the power of selection in the presence and absence of mistranslation"""

import numpy as np
import evolution as evo
import multiprocessing as mp

def random_choice(n, fitdict):
    """From a set of amino acid sequences, pick n sequences at random, find the
    DNA sequences that encode the chosen polypeptides and pick one DNA sequence
    for each amino acid sequence at random."""
    # Perform some kind of quality check??
    aaseqs = list(fitdict.keys())
    aachoice = np.random.choice(aaseqs, n, replace=False)
    ntchoice = []
    for aa in aachoice:
        ntseqs = evo.reverse_translate(aa, evo.codontable)
        ntchoice.append(np.random.choice(ntseqs))
    return ntchoice

def get_competitor_sets(ngeno, fitdict, all=False):
    # Choose random focal genotypes
    wtseqs = random_choice(ngeno, fitdict)
    # Find all neighbours of focal genotypes (not including stop codons)
    # and place in list of all combinations that are to be simulated
    if all:
        comp = [ [wt, mt] for wt in wtseqs for mt in evo.find_neighbours(wt)]
    else:
        comp = []
        for wt in wtseqs:
            # find neighbours
            neigh = evo.find_neighbours(wt)
            # remove those for which there is no data
            neigh = [mt for mt in neigh if evo.translate(mt, evo.transltable) in fitdict.keys()]
            # sample one neighbour at random
            mt = np.random.choice(neigh)
            comp.append([wt, mt])
    return comp

def remove_mutants_nodata(comp, fitdict):
    return [row for row in comp if evo.translate(row[1], evo.transltable) in fitdict.keys()]

def save_competitors(comp, fname):
    with open(fname, "w") as f:
        for pair in comp:
            f.write("{},{}\n".format(pair[0], pair[1]))

def read_competitors(fname):
    comp = []
    with open(fname, "r") as f:
        for line in f.readlines():
            comp.append(line.strip().split(","))
    return comp

def save_results(results, directory):
    wt, mt = results[:2]
    with open(directory+"results_{}_{}.txt".format(wt, mt), "w") as f:
        # Put into a single line
        f.write(",".join([str(v) for v in results]))

def get_fix_prob(N, pr, fitfile, compfile, compidx, threshold, directory):
    # fitfile: File containing the fitness dictionary
    # compfile: File containing the competitors
    # compidx: Which competitions to perform in this run
    fitdict = evo.read_fitness_dictionary(fitfile)
    comp = read_competitors(compfile)
    for i in compidx:
        wt,mt = comp[i]
        f, s_mis, psums, Ne, u_p_mis = evo.mistranslation_fixation_prob(N, pr, wt, mt, fitdict, threshold)
        fwt, fmt, s, u_p = evo.no_mistranslation_fixation_prob(N, wt, mt, fitdict)
        res = [wt, mt, fwt, fmt, f[0][0], f[0][1], f[1][0], f[1][1], s, s_mis, N, Ne, u_p,
                    u_p_mis, pr, psums[0], psums[1]]
        save_results(res, directory)
    return True

def parallel_fixation_probability(N=1000, pr=100, threshold=1e-9, startidx=0,
    endidx=3375, compnum=10, fitfile="../data/gb1_fitdict.txt",
    compfile="../data/gb1_competitors.txt", directory="../results/gb1_selection_power/"):
    pool = mp.Pool(processes=mp.cpu_count()-1)
    workers = []
    for i in range(startidx, endidx, compnum):
        p = pool.apply_async(get_fix_prob,
            args=(N, pr, fitfile, compfile, list(range(i,i+compnum)), threshold,
                directory,))
        workers.append(p)
    res = []
    for p in workers:
        res.append(p.get())
    if not all(res):
        raise RuntimeError("Something went wrong!")

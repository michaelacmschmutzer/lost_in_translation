#!/usr/bin/env python3

import os
import sys
import numpy as np
import sqlite3 as sql
import evolution as evo

"""Simulate an adaptive walk through a fitness landscape"""


    # Two options are open:
    # 1. take the steepest step until no futher increases in fitness are
    # available among neighbours - > No, because I want to see "escapes" from local fitness
    # optima
    # 2. Calculate the probability of fixation of all neighbours (as in 1), and
    # then choose the mutation to be fixed randomly. This included the possibility
    # that no mutation is fixed. Every step then is the period between the
    # emergence of mutations, which I assume to be so rare that in the intermediate
    # period the mutation is either fixed or lost through drift or negative
    # selection -> Better, avoid conflict from clonal interference. Can this be
    # calculated in number of generations till fixation?
    # -> but then, would this model then not also have to simulate events where
    # mutations are lost? Yes

def choose_starting_genotypes_gb1(gb1, percentile=10):
    # Do not include proteins that have zero fitness! These are deemed "inviable"
    viable_all = np.array([v for v in gb1.values() if v > 0])
    threshold = np.percentile(viable_all, percentile) # calculate the 10th percentile
    sequences = [k for k,v in gb1.items() if v <= threshold and v > 0] # get all beneath 10%
    aaslct = np.random.choice(sequences, 10000, replace=False)
    ntslct = []
    for aa in aaslct:
        ntseqs = evo.reverse_translate(aa)
        ntslct.append(np.random.choice(ntseqs))
    return ntslct

def choose_starting_genotypes_pard3(landscape, percentile=10):
    # must start by selecting genotype sequences, as there are only 6533 viable
    # genotypes in landscape
    genotypes = {}
    for k,v in landscape.items():
        ntseqs = evo.reverse_translate(k, single=False)
        for seq in ntseqs:
            genotypes[seq] = v
    viable_all = np.array([v for v in genotypes.values() if v > 0])
    threshold = np.percentile(viable_all, percentile)
    sequences = [k for k,v in genotypes.items() if v <= threshold and v > 0]
    ntslct = np.random.choice(sequences, 10000, replace=False)
    return ntslct

def potential_mutations(N, ntseq, fit, landscape, name):
    neigh = evo.find_neighbours(ntseq)
    muts = {} # Sequences and their fixation probabilities
    fits = {ntseq : fit} # Sequences and their fitnesses
    if type(landscape) == sql.Cursor:
        nfits = {}
        landscape.execute("SELECT * FROM {} WHERE seq IN ({})".format(name, ",".join("?"*len(neigh))), neigh)
        out = landscape.fetchall()
        for (seq,fmean,fvar,psum) in out:
            nfits[seq] = [fmean, fvar]
    for nt in neigh:
        # Get the fitness of the neighbours, and from that their probs of fixation
        if type(landscape) == dict:
            aaseq = evo.translate(nt)
            if aaseq in landscape.keys():
                fits[nt] = landscape[aaseq]
                u_p = evo.no_mistranslation_fixation_prob(N, ntseq, nt, landscape)[3]
            else:
                u_p = 0 # simply unknown
        elif type(landscape) == sql.Cursor:
            if nt in nfits.keys():
                nfit = nfits[nt]
                fits[nt] = nfits[nt]
                Ne, u_p, s = evo.mistranslation_fixation_prob_from_fit(N, [fit, nfit])
            else:
                u_p = 0
        else:
            raise RuntimeError("Unknown landscape type")
        muts[nt] = u_p
    return muts, fits

def mutation_fixation(wtseq, mutations):
    # Mutation occurs randomly at with equal probability
    mtseq = np.random.choice(list(mutations))
    # Mutation is either fixed or lost
    prob = mutations[mtseq]
    fix = np.random.choice([True, False], p=[prob, 1-prob])
    # Return the sequence that has fixed
    if fix:
        return mtseq
    else:
        return wtseq

def adaptive_walk(initgen, landscape, steps=int(1e6), N=int(1e4), name=None):
    walk = np.repeat("X"*len(initgen), steps)
    fitness = np.repeat(-1., steps)
    walk[0] = initgen
    ntseq = initgen
    # Read in the starting fitness
    if type(landscape) == dict:
        aaseq = evo.translate(initgen)
        fit = {initgen : landscape[aaseq]}
        fitness[0] = fit[initgen]
    elif type(landscape) == sql.Cursor:
        landscape.execute("SELECT * FROM {} WHERE seq = ? ".format(name), (initgen,))
        out = landscape.fetchone()
        fit = {initgen : [out[1], out[2]]}
        fitness[0] = fit[initgen][0]
    else:
        raise RuntimeError("Unknown landscape type")
    # Simulate mutation-fixation events
    for i in range(1,steps):
        muts, fit = potential_mutations(N, ntseq, fit[ntseq], landscape, name)
        ntseq = mutation_fixation(ntseq, muts)
        walk[i] = ntseq
        if type(landscape) == dict:
            fitness[i] = fit[ntseq]
        else:
            fitness[i] = fit[ntseq][0] # Record only the mean fitness
    return walk, fitness

def expand(cond):
    traj = np.empty(np.sum([int(v) for v in cond[:,1]]), dtype=cond.dtype)
    pos = 0
    for i in range(cond.shape[0]):
        rep = int(cond[i,1])
        newvals = np.repeat(cond[i,0], rep)
        traj[pos:pos+rep] = newvals
        pos += rep
    return traj

def condense(traj):
    cond = []
    fit = traj[0]
    count = 1
    for i in range(1, traj.shape[0]):
        curfit = traj[i]
        if curfit == fit:
            count += 1
        else:
            cond.append([fit, count])
            count = 1
            fit = curfit
    cond.append([fit, count])
    return np.array(cond)

def main(ID, N, landscape_file = "gb1_1.db", start_seqs="adaptive_walk_slct_low.txt", results_dir="../results/adaptive_walk/"):
#    clusterdir = "/home/mschmutzer/mistranslation/"
    clusterdir = "../"
    # Starting sequence
    slct = np.loadtxt(clusterdir+"data/"+start_seqs, dtype=str)
    initgen = slct[ID]
    del slct # No longer needed
    # Load landscape
    if landscape_file.endswith(".db"):
        mistranslation = "True"
        name = landscape_file.split("_")[0]
        db = sql.connect(clusterdir+"data/" + landscape_file)
        landscape = db.cursor()
    else:
        mistranslation = "False"
        name = None
        landscape = evo.read_fitness_dictionary(clusterdir+"data/"+landscape_file)
    # Simulate
    walk, fitness = adaptive_walk(initgen, landscape, int(1e5), N, name=name)
    # condense (saves disk space)
    fitcond = condense(fitness)
    walkcond = condense(walk)
    # save results as seq _ landscape _ popsize .txt
    fname = "_".join([initgen, mistranslation, "1e{}".format(int(np.log10(N)))])
    np.savetxt(results_dir+fname+"_fitness.txt", fitcond)
    np.savetxt(results_dir+fname+"_walk.txt", walkcond, fmt="%s")


if __name__ == "__main__":
    ID = int(os.environ["SLURM_ARRAY_TASK_ID"])
#    ID = sys.argv[1]
    # Landscape to use. Mistranslation or not?
    mistranslation = sys.argv[1]
    # Population size
    N = int(sys.argv[2])
    main(ID, mistranslation, N)

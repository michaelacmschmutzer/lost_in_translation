#!/usr/bin/env python3

"""For use on cluster: Calculate the fitness of every genotype and save in
separate files"""

import evolution as evo
import numpy as np
import sys
import os

def get_aaseqs(index, fname="/home/mschmutzer/mistranslation/data/pard3e2seqs.txt"):
    seqs = []
    with open(fname, "r") as f:
        for line in f.readlines():
            seqs.append(line.strip())
    if (index+1) * 20 < len(seqs):
        aas = seqs[index*20:(index+1)*20]
    else:
        aas = seqs[index*20:len(seqs)]
    return aas

def calculate_fitness(ntseq, fitdict, prcount):
    prob, fits = evo.expected_mistranslation(ntseq, fitdict)
    prob, fits = evo.prune_probabilities(prob, fits, threshold=1e-9)
    fmean, fvar = evo.expected_fitness_moments(prcount, prob, fits)
    return fmean, fvar, np.sum(prob)

if __name__ == "__main__":
    ID = int(os.environ["SLURM_ARRAY_TASK_ID"])
    aaseqs = get_aaseqs(ID)# Pass the polypeptide sequence
    prcount = int(sys.argv[1]) # How many proteins per cell?
    directory = sys.argv[2] # Where to save results?
    fitdir = sys.argv[3] # Where is the fitness data?
    fitdict = evo.read_fitness_dictionary(fitdir)
    for aaseq in aaseqs:
        # Get all nucleotide sequences
        nts = evo.reverse_translate(aaseq, single=False)
        # Calculate fitnesses
        for nt in nts:
            fmean, fvar, probsum = calculate_fitness(nt, fitdict, prcount)
            np.savetxt(directory+nt+".txt", [fmean, fvar, probsum])

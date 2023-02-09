#!/usr/bin/env python3

import numpy as np
import networkx as nx
import simplejson as json
from itertools import combinations, product


#directory = "/home/mschmutzer/mistranslation/"
directory = "../"

###############################################################################
# Functions for loading and other preparations
###############################################################################

def read_rates(fname):
    misrates = {}
    with open(fname, "r") as f:
        for line in f.readlines():
            items = line.split(",")
            c = items[0]
            aa = items[1]
            r = [float(n) for n in items[2:4]]
            r.append(int(items[-1]))
            if c in misrates.keys():
                misrates[c][aa] = r
            else:
                misrates[c] = {aa: r}
    return misrates

def read_landscape(fname):
    """Read in a network representing a fitness landscape from a json file"""
    G = nx.Graph()
    d = json.load(open(fname))
    G.add_nodes_from(d['nodes'])
    G.add_edges_from(d['edges'])
    return G

def read_codon_table(fname="../data/codon_table.txt"):
    with open(fname, "r") as f:
        lines = f.readlines()
    codon_table = {} # Find codons given amino acid
    trnsl_table = {} # Find amino acid given codon
    for line in lines:
        if not line.startswith("Inf"):
            cont = line.split(" ")
            if cont[1] in codon_table.keys():
                codon_table[cont[1]].append(cont[0])
            else:
                codon_table[cont[1]] = [cont[0]]
            trnsl_table[cont[0]] = cont[1]
    return codon_table, trnsl_table

def read_fitness_dictionary(fname):
    fitdict = {}
    with open(fname, "r") as f:
        for line in f.readlines():
            k,v = line.split(",")
            fitdict[k] = float(v)
    return fitdict

def write_fitness_dictionary(fname, fitdict):
    with open(fname, "w") as f:
        for k,v in fitdict.items():
            line = ",".join([k,str(v)])
            f.write(line+"\n")

def get_amino_acids(codontable):
    amino_acids = list(codontable.keys())
    amino_acids.remove("*")
    return amino_acids

def read_fitness(fname):
    fit = np.loadtxt(fname)
    return fit

###############################################################################
# Creating and saving fitness dictionary
###############################################################################

def make_fitness_dictionary(fname="../data/gb1_network.json"):
    """Get amino acid fitness relationship into dictionary, takes up much less
    memory than a networkx object"""
    aascape = read_landscape(fname)
    fitdict = {s: aascape.nodes[s]["fitness"] for s in aascape}
    return fitdict

def save_fitness_dictionary(fitdict, fname="../data/fitness_dictionary.txt"):
    """Save fitness dictionary"""
    with open(fname, "w") as f:
        for k,v in fitdict.items():
            f.write("{},{}\n".format(k,v))
    return None

###############################################################################
# Calculating the effects of mistranslation
###############################################################################

def find_neighbours(ntseq):
    """Given a nucleotide sequence, find all its single-mutation neighbours that
    do not contain a premature stop codon"""
    neigh = []
    for i in range(len(ntseq)):
        for nt in nucleotides:
            # Only 3 possible alternative nt for each position
            if nt != ntseq[i]:
                mtseq = ntseq[:i] + nt + ntseq[i+1:]
                # Make sure that mtseq doesn't include a stop codon
                mtaa = translate(mtseq, transltable)
                if "*" not in mtaa:
                    neigh.append(mtseq)
    return neigh

def expected_mistranslation(ntseq, fitdict, return_seq=False):
    """Calculate the probabilities of expressing a given protein and its
    contribution to fitness"""
    # Perhaps add a threshold to ignore the effect of very rare sequences
    # Get the propabilities of substitution for each amino acid
    # Get the amino_acid sequences and fitnesses of each of the amino acid neighbours
    # Get the substitution probabilites for each codon
    # For now, try only one substitution per sequence
#    proteins = [] # create a list of all possible sequences from (mis)translation,
    probabilities = [] # their associated probabilities,
    fitnesses = [] # and their fitnesses
    aaprobs = {p : { aa : 0 for aa in amino_acids} for p in range(len(ntseq)//3)}
    if return_seq:
        polypeptides = []
    for i in range(0, len(ntseq), 3):
        # get codon and mistranslation rates for each aa from codon
        codon = ntseq[i:i+3]
        rates = misrates[codon]
        aa = list(rates.keys())
        probs = [r[0] for r in rates.values()]
        # Check if probs sum to less than one. If not, then I have a problem
        if sum(probs) > 1:
            raise RuntimeError("Sum of mistranslation probs > 1")
        # does not include cognate aa. Add in prob. of getting cognate aa
        aa.append(transltable[codon])
        probs.append(1-sum(probs)) # probability of not mistranslating
        for j,a in enumerate(aa):
            aaprobs[i//3][a] = probs[j]
    for seq in fitdict.keys():
        p = 1# probability of sequence
        for i,aa in enumerate(seq):
            p *= aaprobs[i][aa]
#        pseq = "".join(seq)
        if p > 0: # and pseq in fitdict.keys():
#            proteins.append(pseq)
            probabilities.append(p)
            fitnesses.append(fitdict[seq])
            if return_seq:
                polypeptides.append(seq)
    # the result is a summary of all possible protein fitnesses and the probability of observing
    # that fitness
    if return_seq:
        return np.array(probabilities), np.array(fitnesses), polypeptides
    else:
        return np.array(probabilities), np.array(fitnesses)

def prune_probabilities(probs, fits, threshold=1e-9):
    """Remove proteins that have very low probability of being produced"""
    new_probs = probs[probs >= threshold]
    new_fits = fits[probs >= threshold]
    # Make sure that all probabilities sum to one. Or not necessary?
    return new_probs, new_fits

def expected_fitness_moments(prcount, probs, fits):
    """Calculate the expected fitness and variation given the 'fitnesses' and
    probabilities from expected_mistranslation"""
#    rv = multinomial(prcount, probs)
    # expected mean individual fitness
    w_mean = np.sum(probs * fits)
    # Calculate covariance piece by piece (less memory intensive)
    varsum = 0
    for i in range(len(probs)):
        for j in range(i, len(probs)):
            if i == j: # variance
                varsum += probs[i] * (1- probs[i]) * fits[i]**2 * 1/prcount
            else: # covariance
                varsum -= 2 * probs[i] * probs[j] * fits[i] * fits[j] * 1/prcount
    return w_mean, varsum

# fitness of an an individual drawn at random
# produced = np.random.multinomial(proteinnum, probabilities)
# fit = np.sum(np.array(fitnesses) * produced) / proteinnum
# return fit

def mistranslation_fixation_prob_from_fit(N, fitness):
    # Determine effective population size using the wt
    fmean, fvar = fitness[0]
    fnoise = fvar / fmean**2
    Ne = popsize_reduction(N,fnoise)
    # Calculate predicted fixation probabilities
    s = fitness[1][0] / fitness[0][0] - 1
    if s != 0:
        u_p = kimura_fixation_prob(Ne, s, p=1/N)
    else:
        u_p = neutral_fixation_probability(N)
    return Ne, u_p, s

def mistranslation_fixation_prob(N, pr, wtseq, mtseq, fitdict, threshold=1e-15):
    """Calculate the mean and variation in fitness, the effective population
    size and the fixation probability of the mutant when invading a population
    carrying the wild-type sequence in the presence of mistranslation."""
    # Calculate for each genotype the mean fitness and variation
    fitness = []
    psums = []
    for seq in [wtseq, mtseq]:
        probs, fits = expected_mistranslation(seq, fitdict)
        probs, fits = prune_probabilities(probs, fits, threshold)
        fmean, fvar = expected_fitness_moments(pr, probs, fits)
        fitness.append([fmean, fvar])
        psums.append(np.sum(probs))
    Ne, u_p, s = mistranslation_fixation_prob_from_fit(N, fitness)
    return fitness, s, psums, Ne, u_p

def no_mistranslation_fixation_prob(N, wtseq, mtseq, fitdict):
    """Predict the fixation probability in the absence of mistranslation"""
    # Get fitnesses
    f_wt = fitdict[translate(wtseq, transltable)]
    f_mt = fitdict[translate(mtseq, transltable)]
    # Calculate selection coefficient
    if f_wt == 0: # This happens! Complete nonsense, of course
        s = np.nan
        u_p = 1
    else:
        s = f_mt/f_wt - 1
        # calculate fixation probability
        if s != 0:
            u_p = kimura_fixation_prob(N, s, p=1/N)
        else:
            u_p = neutral_fixation_probability(N)
    return f_wt, f_mt, s, u_p

def stochastic_mistranslation(N, prcount, ntseq, fitdict):
    """Return fitnesses for every individual in population"""
    proteins=np.empty([N,prcount,len(ntseq)//3], dtype="str")
    for c in range(0, len(ntseq), 3):
        codon = ntseq[c:c+3]
        rates = misrates[codon]
        aa = list(rates.keys())
        probs = [r[0] for r in rates.values()] # does not include cognate aa
        aa.append(transltable[codon])
        probs.append(1-sum(probs)) # probability of not mistranslating
        # What if, due to error, probabilities of noncognate amino acids
        # have a sum larger than one? -> need to test for this
        proteins[:,:,c//3] = np.random.choice(aa, size=(N,prcount), replace=True, p=probs)
    # Look up sequences in proteins and get fitnesses
    fitnesses = np.empty([N,])
    missing = 0
    for i in range(proteins.shape[0]):
        fit = []
        for p in range(proteins.shape[1]):
            seq = "".join(proteins[i,p])
            if seq in fitdict:
                fit.append(fitdict[seq])
            else:
                missing += 1 # Ignore for now. How to deal with missing data?
        # Take the average fitness
        # To explore different weightings (eg low fitness proteins have larger
        # effects) use np.average with weights
        fitnesses[i] = np.mean(fit)
    # repeat for all individuals
    if missing > 0:
        print("{} proteins missing".format(missing))
    return fitnesses

def find_distribution_num_mistr(aaseq, polypeptides, probs):
    """Find the probability of having a given number of mistranslated aa in
    the polypeptide sequence"""
    misprob = {i: [] for i in range(len(aaseq)+1)}
    for i in range(probs.shape[0]):
        diff = hamming(aaseq, polypeptides[i])
        misprob[diff].append(probs[i])
    return misprob

def write_gexf(landscape, fname):
    """For gephi plotting"""
    nx.write_gexf(landscape, fname)
    return None

def selection_coefficients(landscape):
    """Calculate the selection coefficient between two neighbouring genotypes.
    Calculate only once for each pair"""
    selcoeff = []
    selsmooth = []
    for edge in landscape.edges:
        # this way counts each edge only once, but arbitrarily, for networkx stores each
        # edge with the lower node id first and then the higher node id second
        # Therefore store as an absolute value
        s = landscape.nodes[edge[0]]["fitness"] - landscape.nodes[edge[1]]["fitness"]
        if not np.isnan(s): # Get rid of nans. Don't know what is going on so ignore
            s_smoothed = landscape.nodes[edge[0]]["fitness_smoothed"] - landscape.nodes[edge[1]]["fitness_smoothed"]
            selcoeff.append(abs(s))
            selsmooth.append(abs(s_smoothed))
            landscape.edges[edge]["scoeff"] = abs(s)
            landscape.edges[edge]["scoeff_smoothed"] = abs(s_smoothed)
    return selcoeff, selsmooth, landscape

def switch_selection_sign(selcoeff, selsmooth):
    """Find those selection coefficients that changed sign (i.e. became positive
    instead of negative) and return magnitude of change"""
    sign_switch = []
    for i,j in zip(selcoeff, selsmooth):
        if np.sign(i) != np.sign(j) and not np.isnan(i) and not np.isnan(j):
            sign_switch.append([i,j,j-i])
    return sign_switch

def kimura_fixation_prob(N, s, p=0.001):
#    u = (2 * s) / (1 - np.exp(- 2 * N * s))
# directly from: https://gsejournal.biomedcentral.com/track/pdf/10.1186/1297-9686-17-3-351?site=gsejournal.biomedcentral.com
# see also the fixation equation in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5806465/?report=reader
    u = (1 - np.exp(- 2 * N * p * s)) / (1 - np.exp(- 2 * N * s))
    return u

def neutral_fixation_probability(N):
    return 1/N

def popsize_reduction(N, w_noise):
    """Reduction in effective population size according to Wang & Zhang (2011).
    The w_noise is the CV^2 """
    # w_noise is CV**2
    N_e = N / (1 + w_noise)
    return N_e

def hamming(u, v):
    if len(u) != len(v):
        raise ValueError("Sequences must be of same length")
    return sum(x!=y for x,y in zip(u,v))

###############################################################################
# Set global variables

codontable, transltable = read_codon_table(directory+"data/codon_table.txt")
amino_acids = get_amino_acids(codontable)
nucleotides = ["A", "T", "G", "C"]
misrates = read_rates(directory+"data/misrates_intpol.csv")

##############################################################################
# Functions for translation and reverse translation

def reverse_translate(aasequence, codon_table=codontable, single=False):
    """Return all the possible DNA sequences that encode an amino acid sequence"""
    if single:
        ntseq = codon_table[aasequence[0]][0]
        for aa in aasequence[1:]:
            ntseq += codon_table[aa][0] # Could also make this a random choice to avoid bias
        ntseqs = [ntseq]
    else:
        ntseqs = codon_table[aasequence[0]]
        for aa in aasequence[1:]:
            codons = codon_table[aa]
            new_ntseqs = []
            for i,sq in enumerate(ntseqs):
                for c in codons:
                    newsq = sq + c
                    new_ntseqs.append(newsq)
            ntseqs = new_ntseqs
    return ntseqs

def translate(ntseq, transltable=transltable):
    aa = [transltable[ntseq[c:c+3]] for c in range(0,len(ntseq),3)]
    return "".join(aa)

##############################################################################
# How robust are the results to missing data?

def all_neighbours(gb1):
    # find a sequence with all its neihbours present
    ncount = 0
    while ncount != 80:
        aaseq = np.random.choice(list(gb1.keys()))
        ncount = 0
        for i in range(len(aaseq)):
            for aa in amino_acids:
                nseq = aaseq[:1] + aa + aaseq[i+1:]
                if nseq in gb1.keys():
                    ncount += 1
    return aaseq

def missing_data_effects(gb1, aaseq = "ILGL", pr=1):
    altfit = []
    ntseq = reverse_translate(aaseq, single=True)[0]
    for i in range(len(aaseq)):
        for aa in amino_acids:
            if aa != aaseq[i]:
                nseq = aaseq[:i] + aa + aaseq[i+1:]
                # make a landscape missing the neighbour in question
                gb1_alt = {k:v for k,v in gb1.items() if k != nseq}
                prob,fit = expected_mistranslation(ntseq, gb1_alt)
                prob,fit = prune_probabilities(prob, fit)
                fmean, fvar = expected_fitness_moments(pr, prob, fit)
                altfit.append([i, aa, ntseq, fmean, fvar])
    return altfit


# A way in which to code population dynamics (N*mu << 1)
# First: Calculate probability and time (generations) to mutation
# Second: Calculate probability of and time to fixation/extinction
# Third: If fixed, move population to new genotype. Otherwise, start again
# Calculating the number of generations is important because it is the only way to
# know whether a population got "stuck" in a local optimum... if I were to force the
# population to always chose a mutation as the next step the generation time spend at
# a given genotype could not be measured

#!/usr/bin/env python3

"""Find neutral networks in genotype landscapes given a set of starting
protein sequences. Count the number of peaks and determine epistasis"""

import os
import copy
import pickle
import numpy as np
import evolution as evo
import pandas as pd
import seaborn as sns
import collections
from itertools import combinations
from tqdm import tqdm
import sqlite3 as sql
import matplotlib.pyplot as plt

# pick X sequences at random and determine on what neutral network they are on

def make_index2_database(dbfile):
    # Make an index on fmeans to accelerate sorting
    db = sql.connect(dbfile)
    cur = db.cursor()
    cur.execute("CREATE INDEX Idx2 on gb1 (fmean)")
    db.commit()
    db.close()

def find_neutral_neighbours(landscape, seq, Ne, name="gb1", return_all=False):
    nearly_neutral = []
    if return_all:
        non_neutral = []
        mut_kind = []
    allseqs = evo.find_neighbours(seq)
    # Get the fitnesses of the focal genotype and its neighbours
    if type(landscape) == collections.OrderedDict or type(landscape) == dict:
        # Because these values have no mistranslation, fitness = fmean, fvar is zero
        neighs = {s : [landscape[evo.translate(s)],0,1] for s in allseqs if evo.translate(s) in landscape.keys()}
        fmean = landscape[evo.translate(seq)]
        fvar = 0 # No mistranslation
    else:
        allseqs.append(seq)
        landscape.execute("SELECT * FROM {} WHERE seq IN ({})".format(name, ",".join("?"*len(allseqs))), allseqs)
        out = landscape.fetchall()
        neighs = {row[0]: row[1:] for row in out if not row[0] == seq}
        fmean,fvar = [row[1:3] for row in out if row[0] == seq][0]
    for nseq,(nfmean, nfvar, npsum) in neighs.items():
        if fmean != 0 and nfmean != 0: # This can happen with the empirical/no
            #  mistranslation dataset
            # Calculate the selection coefficient from the absolute fitnesses
            s0 = nfmean / fmean - 1
            # also calculate the selection coefficient the other way
            s1 = fmean / nfmean - 1
            # Effective population size of the population the mutation is invading
            fnoise = fvar / fmean**2
            Ne_adj = evo.popsize_reduction(Ne,fnoise)
            nfnoise = nfvar / nfmean**2
            nNe_adj = evo.popsize_reduction(Ne,nfnoise)
            # Use the direction in which selection is stronger
            # Use absolute because selection coefficient can be negative
            slcstr = abs(Ne_adj * s0)
            nslcstr = abs(nNe_adj * s1)
            idx = np.argmax([slcstr, nslcstr])
            sel = [slcstr, nslcstr][idx]
            if sel < 0.25:
                nearly_neutral.append(nseq)
            elif return_all:
                if sel < 0.25:
                    mut_kind.append("neutral")
                elif s0 > 0:
                    non_neutral.append(nseq)
                    mut_kind.append("beneficial")
                else:
                    non_neutral.append(nseq)
                    mut_kind.append("deleterious")
    if return_all:
        return nearly_neutral, non_neutral, mut_kind
    else:
        return nearly_neutral

def find_neutral_neighbours_disentangle(landscape, mislandscape, seq, Ne, name="gb1", consider="Ne"):
    """Find neutral networks when either the smoothing or the Ne effects of
    mistranslation are disabled. Consider either 'Ne' or 'smooth' """
    nearly_neutral = []
    allseqs = evo.find_neighbours(seq)
    allseqs.append(seq)
    if consider == "both":
        return find_neutral_neighbours(landscape, seq, Ne)
    elif consider == "Ne":
        # Adjust Ne with fvar from mislandscape, fmean from landscape
        mislandscape.execute("SELECT seq,fvar FROM {} WHERE seq IN ({})".format(name,",".join("?"*len(allseqs))), allseqs)
        out = mislandscape.fetchall()
        fvar = {row[0] : row[1] for row in out}
        fmean = {s: landscape[evo.translate(s)] for s in allseqs if evo.translate(s) in landscape.keys()}
    elif consider == "smooth":
        # Keep Ne, get fmean from mislandscape
        mislandscape.execute("SELECT seq,fmean FROM {} WHERE seq IN ({})".format(name,",".join("?"*len(allseqs))), allseqs)
        out = mislandscape.fetchall()
        fvar = {s : 0 for s in allseqs}
        fmean = {row[0]: row[1] for row in out}
    else:
        raise RuntimeError("consider must be either both, smooth or Ne")
    for s in allseqs:
        if s != seq and s in fmean.keys(): # neighbours. It can happen that no entry for s exists in the landscape
            if fmean[seq] != 0 and fmean[s] != 0: # This can happen with the empirical/no mistranslation
                # Calculate the selection coefficient from the absolute fitnesses
                s0 = fmean[s] / fmean[seq] - 1
                # also calculate the selection coefficient the other way
                s1 = fmean[seq] / fmean[s] - 1
                # Effective population size of the population the mutation is invading
                fnoise = fvar[seq] / fmean[seq]**2
                Ne_adj = evo.popsize_reduction(Ne,fnoise)
                # effective population size the other way round
                nfnoise = fvar[s] / fmean[s]**2
                nNe_adj = evo.popsize_reduction(Ne,nfnoise)
                # Use the direction in which selection is stronger
                slcstr = abs(Ne_adj * s0)
                nslcstr = abs(nNe_adj * s1)
                idx = np.argmax([slcstr, nslcstr])
                sel = [slcstr, nslcstr][idx]
                if sel < 0.25:
                    nearly_neutral.append(s)
    return nearly_neutral

def find_neutral_network(startseq, landscape, Ne=1e6, mislandscape=None, name="gb1", consider=None):
    # Given a starting sequence, find all the sequences that are on the same
    # neutral network as it
    neutral = {}
    neutral[startseq] = False # Has this sequence's neighbours been assessed yet?
    # Continue until no new neutral neighbours have been found. The neutral network
    # is then complete
    while not all([v == True for v in neutral.values()]):
        new_seqs = []
        for k in neutral.keys():
            if not neutral[k]:
                neutral[k] = True
                if consider:
                    nn = find_neutral_neighbours_disentangle(landscape, mislandscape, k, Ne, name, consider)
                else:
                    nn = find_neutral_neighbours(landscape, k, Ne, name)
                new_seqs.extend(nn)
        for seq in new_seqs:
            # Check if this sequence has been evaluated already. If not, its
            # neighbours will be checked in the next iteration of the while loop
            if not seq in neutral.keys():
                neutral[seq] = False
    return [seq for seq in neutral.keys()]

def map_neutrality(ntseqs, landscape, Ne=1e6, mislandscape=None, name="gb1", consider=None):
    """Find nearly neutral networks for a set of starting nucleotide sequences"""
    networks = {} # sequence: network ID
    id = 0
    for nt in tqdm(ntseqs):
        # Check if sequence is already known
        if not nt in networks.keys():
            # Find its neutral network
            neut = find_neutral_network(nt, landscape, Ne, mislandscape, name, consider)
            for seq in neut:
                networks[seq] = id
            # Next network gets a new id
            id += 1
    return networks

def choose_nucleotide_sequences(gb1, size=100):
    # Choose random polypeptide sequences that are more than one amino acid
    # substitution away from one another, and generate their neighbours as well
    # Only get sequences that have a fitness greater than zero
    aaseqs = list(np.random.choice([k for k,v in gb1.items() if v > 0], size, replace=False))
    ntseqs = []
    for aa in aaseqs:
        nt = evo.reverse_translate(aa, single=True)[0]
        ntseqs.append(nt)
    return ntseqs

def find_landscape_neutral_networks(landscape, Ne=1e6, maxlen=13370751, name="gb1", networks=None, non_neutral_edges=None):
    network_kind = {}
    if networks and non_neutral_edges:
        netnum = max([v for v in networks.values()]) + 1
    else:
        networks = {}
        non_neutral_edges = {}
        netnum = 0
    if type(landscape) == dict: # Order the landscape from fittest to least fit genotype
        ordlandscape = collections.OrderedDict()
        for k,v in sorted(landscape.items(), key = lambda item: item[1], reverse=True):
            ordlandscape[k] = v
        landscape = ordlandscape
    while len(networks) < maxlen:
        # Map all above a given fitness
        # Get out a sequence to start looking for the next peak, go from
        # high to low fitness
        found = True
        if type(landscape) == collections.OrderedDict:
            for aaseq in landscape.keys():
                ntseq = evo.reverse_translate(aaseq)
                for nt in ntseq:
                    if not nt in networks.keys():
                        found = False
                        seq = nt
                        lowestfitness = landscape[aaseq]
                        break
                if not found:
                    break
                    # This sequence will be the starting point
                    # for exploring the next fitness peak
        else:
            found = True
            landscape.execute("SELECT seq,fmean FROM {} ORDER BY fmean DESC".format(name))
            while found:
                seq,fmean = landscape.fetchone()
                if not seq in networks.keys():
                    found = False # This sequence will be the starting point
                    # for exploring the next fitness peak
                    lowestfitness = fmean
        neutral = {}
        non_neutral_edges[netnum] = []
        mut_kinds = set([])
        neutral[seq] = False # Has this sequence's neighbours been assessed yet?
        # Continue until no new neutral neighbours have been found. The neutral network
        # is then complete
        print("\r", end='', flush=True)
        print("NumSeq: {} and Peak num: {}   ".format(0,netnum), end='', flush=True)
        while not all([v == True for v in neutral.values()]):
            new_seqs = []
            new_edges = [] # New non-neutral edges
            for k in neutral.keys():
                if not neutral[k]:
                    neutral[k] = True
                    nn, not_nn, mtkind = find_neutral_neighbours(landscape, k, Ne, return_all=True, name=name)
                    new_seqs.extend(nn)
                    new_edges.extend(not_nn)
                    for v in mtkind:
                        mut_kinds.add(v)
            for seq in new_edges:
                non_neutral_edges[netnum].append(seq)
            for seq in new_seqs:
                # Check if this sequence has been evaluated already. If not, its
                # neighbours will be checked in the next iteration of the while loop
                if not seq in neutral.keys():
                    neutral[seq] = False
            print("\r", end='', flush=True)
            print("NumSeq: {} and Peak num: {}   ".format(len(neutral), netnum),
                end='', flush=True)
        for k in neutral.keys():
            networks[k] = netnum
        if "deleterious" in mut_kinds and "beneficial" in mut_kinds:
            network_kind[netnum] = "saddle"
        elif "deleterious" in mut_kinds:
            network_kind[netnum] = "peak"
        else:
            network_kind[netnum] = "trough"
        netnum += 1
    return networks, non_neutral_edges, network_kind

def save_networks(fname, networks):
    with open(fname, "wb") as f:
        pickle.dump(networks, f)

def read_networks(fname):
    with open(fname, "rb") as f:
        network = pickle.load(f)
    return network

def restructure_networks(networks):
    neutral = {}
    for k,v in networks.items():
        if not v in neutral.keys():
            neutral[v] = [k]
        else:
            neutral[v].append(k)
    return neutral

def get_network_sizes(neutral):
    network_sizes = []
    for k, v in neutral.items():
        network_sizes.append(len(v))
    return network_sizes

def plot_network_size_distributions(directory):
    flist = os.listdir(directory)
    for f in flist:
        if ".pkl" in f:
            networks = read_networks(directory+f)
            neutral = restructure_networks(networks)
            sizes = get_network_sizes(neutral)
            # plot distributions
            plt.hist(np.log10(sizes), bins=20, range=[0, 3.5], log=True)
            plt.ylim(0.5,1e4)
            plt.xlim(0,3.5)
            plt.yscale("log")
            plt.xlabel(r"Network size (log$_{10}$)")
            plt.ylabel("Number of observations")
            plt.savefig(directory+f.split(".")[0]+".pdf")
            plt.close()


def link_neutral_networks(networks, nonntredges):
    network_links = {}
    for netw,edges in nonntredges.items():
        netwids = []
        for seq in edges:
            if seq in networks.keys():
                netwids.append(networks[seq])
            else:
                netwids.append(np.nan)
        neighs = collections.Counter(netwids)
        for neighnw,count in neighs.items():
            edge = tuple(sorted([netw, neighnw]))
            # Check if the number of links is the same both ways
            if edge in network_links.keys():
                if not network_links[edge] == count:
                    raise RuntimeError("Old and new counts not the same")
            else:
                network_links[edge] = count
    return network_links

def get_mean_network_fitnesses(landscape, neutral):
    mean_fitness = {}
    for k,seqs in neutral.items():
        if type(landscape) == dict:
            fit = []
            for s in seqs:
                fit.append(landscape[evo.translate(s)])
        else:
            landscape.execute("SELECT fmean FROM gb1 WHERE seq IN ({})".format(",".join("?"*len(seqs))), seqs)
            fit = landscape.fetchall()
        mean_fitness[k] = np.mean(fit)
    return mean_fitness

def get_fitness(geno, landscape):
    if type(landscape) == dict:
        polypep = evo.translate(geno)
        if polypep in landscape.keys():
            w = landscape[polypep]
        else:
            w = False
    else:
        landscape.execute("SELECT name FROM sqlite_master WHERE type='table'")
        name = landscape.fetchall()[0][0]
        landscape.execute("SELECT fmean FROM {} WHERE seq = ?".format(name), (geno,))
        out = landscape.fetchone()
        if out:
            w = out[0]
        else:
            w = False
    return w

def calculate_epistasis(w00, w01, w10, w11):
    epi = w00 + w11 - w01 - w10
    # is this reciprocal sign epistasis or not?
    # Differences between fitnesses
    D0001 = w01 - w00
    D0010 = w10 - w00
    D0111 = w11 - w01
    D1011 = w11 - w10
    # There are two paths between the genotypes that differ in two positions
    path1 = abs(D0001 + D1011) < abs(D0001) + abs(D1011)
    path2 = abs(D0010 + D0111) < abs(D0010) + abs(D0111)
    if path1 and path2:
        kind = "reci"
    elif path1 or path2:
        kind = "sign"
    elif epi != 0:
        kind = "magn"
    else:
        kind = "none"
    return epi, kind

def sample_epistasis(startgeno, landscape):
    epistasis = []
    kinds = []
#    misepistasis = []
#    miskinds = []
    # use all genotypes in the landscape
    #startgeno = [geno for geno in landscape.keys()]
    n = len(startgeno[0]) # length of genotype sequence (same for all)
    for geno00 in tqdm(startgeno):
        found = False
        while not found:
            # take two position in the sequence to modify
            pos = np.random.choice(range(n), 2, replace=False)
            # For the two positions, choose a different nucleotide at random
            nts = [np.random.choice([nt for nt in evo.nucleotides if nt != geno00[i]], 1)[0] for i in pos]
            # Construct the three other sequences
            geno10 = geno00[:pos[0]] + nts[0] + geno00[1+pos[0]:]
            geno01 = geno00[:pos[1]] + nts[1] + geno00[1+pos[1]:]
            seq = [nt for nt in geno00]
            seq[pos[0]] = nts[0]
            seq[pos[1]] = nts[1]
            geno11 = "".join(seq)
            # find fitnesses (check if in landscape)
            w00 = get_fitness(geno00, landscape)
            w01 = get_fitness(geno01, landscape)
            w10 = get_fitness(geno10, landscape)
            w11 = get_fitness(geno11, landscape)
            if all([type(w) != bool for w in [w00,w01,w10,w11]]):
                found = True # if in empirical landscape, then also in the other
        # calculate epistasis
        epi, kind = calculate_epistasis(w00, w01, w10, w11)
        epistasis.append(epi)
        kinds.append(kind)
        # same for mistranslation
#        w00 = get_fitness(geno00, mislandscape)
#        w01 = get_fitness(geno01, mislandscape)
#        w10 = get_fitness(geno10, mislandscape)
#        w11 = get_fitness(geno11, mislandscape)
#        epi, kind = calculate_epistasis(w00, w01, w10, w11)
#        misepistasis.append(epi)
#        miskinds.append(kind)
    return epistasis, kinds

def epistasis_landscapes(startgeno, landscapes):
    epistasis = []
    reciprocal = []
    for name,l in landscapes.items():
        epi, kind = sample_epistasis(startgeno, l)
        for v,k in zip(epi, kind):
            epistasis.append([name, v, k])
    epistasis = pd.DataFrame(epistasis, columns=["landscape", "epistasis", "kind"])
    return epistasis

def plot_epistasis_landscapes(epistasis):
    sns.violinplot(x="landscape", y="epistasis", data=epistasis)
    plt.ylabel("Epistasis")
    plt.xlabel("")
    plt.show()

def plot_reciprocal_landscapes(reciprocal):
    sns.barplot(x="landscape", y="reciprocal", data=reciprocal)
    plt.ylabel("Percent reciprocal sign epistasis")
    plt.xlabel("")
    plt.show()


# Load landscape
# Calculate or load from previous steps the fitness with and without mistranslation
# Prune all non-neutral edges
# Identify neutral networks and calculate their size

# Repeat for different population sizes
# Differentiate between effects arising from change in Ne and change in s due
# to mistranslation

    # prune the network using the measured fitnesses (no mistranslation)

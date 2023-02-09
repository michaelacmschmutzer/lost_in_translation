
"""Build landscapes for navigation & visualisation"""

# Landscapes used:
# Melamed et al (2013) Deep mutational scanning of an RRM domain of the
# Saccharomyces cerevisiae poly(A)-binding protein. RNA 19, 1537–1551 (2013).
# This landscape derives from a RNA-recognition motif

# Other landscapes:
# Bendixen et al ribozymes https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000300#sec010
# List of useful landscapes from Lyons et al https://www.nature.com/articles/s41559-020-01286-y?proof=t#data-availability
# SI: https://static-content.springer.com/esm/art%3A10.1038%2Fs41559-020-01286-y/MediaObjects/41559_2020_1286_MOESM1_ESM.pdf

# visualisation in gephi?

import numpy as np
import pandas as pd
import networkx as nx
import simplejson as json
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
from collections import Counter
from evolution import translate, transltable, hamming

def read_codon_table(fname="../data/codon_table.txt"):
    with open(fname, "r") as f:
        lines = f.readlines()
    codon_table = {}
    for line in lines:
        if not line.startswith("Inf"):
            cont = line.split(" ")
            if cont[1] in codon_table.keys():
                codon_table[cont[1]].append(cont[0])
            else:
                codon_table[cont[1]] = [cont[0]]
    return codon_table

def save(G, fname):
    json.dump(dict(nodes=[[n, G.nodes[n]] for n in G.nodes()],
                   edges=[[u, v, G.edges[u, v]] for u,v in G.edges()]),
              open(fname, 'w'), indent=2)

# https://stackoverflow.com/questions/3162909/method-to-save-networkx-graph-to-json-graph/8681020

def load(fname):
    G = nx.Graph()
    d = json.load(open(fname))
    G.add_nodes_from(d['nodes'])
    G.add_edges_from(d['edges'])
    return G

def percentage_coverage(landscape, exp_neigh, mtlrates=None):
    """Given the node degree, the mistranslation bias, and the expected number
    of phenotypic neighbours, calculate degree of completeness"""
    if mtlrates:
        pass
    else:
        coverage = [landscape.degree(n) / exp_neigh for n in landscape.nodes]
    return coverage

def calculate_coverage_position(landscape, seqlen, exp_neigh=20):
    """Calculate how much variation is available for each amino acid position"""
    n_prop = [0 for i in range(seqlen)]
    for node in landscape.nodes:
        neigh = [edge[1] for edge in landscape.edges(node)]
        s_count = {i : set([]) for i in range(seqlen)}
        for n in neigh:
            for i in range(seqlen):
                aa = landscape.nodes[n]["sequence"][i]
                s_count[i].add(aa)
        # Divide by the expected number of neighbours to get proportion of neighbours
        # covered at each sequence position
        n_prop = [len(s_count[i])/exp_neigh + p for i,p in enumerate(n_prop)]
    numnodes = len(landscape.nodes)
    return [n/numnodes for n in n_prop]

def build_rrm_landscape():
    # Single mutations
    rrm_single = pd.read_csv("../data/Melamed_2013_ST2.csv")
    # Double mutations
    rrm_double = pd.read_csv("../data/Melamed_2013_ST5.csv")

    # Make graph
    rrm = {}
    # Add the wild type sequence with a "fitness" (enrichment score) of 1
    wtseq = "".join(rrm_single["Residue"].values)
    rrm[wtseq]= 1
#    amino_acids = rrm_single.columns.to_list()[2:]
    node = 2
    # Add the single mutation data
    for i in range(rrm_single.shape[0]):
        for aa in amino_acids:
            if aa != rrm_single.loc[i, "Residue"]: # These are always NaN
                seq = wtseq[:i] + aa + wtseq[i+1:]
                if not np.isnan(rrm_single.loc[i,aa]):
                    rrm[seq] = rrm_single.loc[i,aa]
#                    rrm.add_edge(wtseq, seq)
#                    node += 1
    # Add the double mutation data
    for i in range(rrm_double.shape[0]):
        seqID = rrm_double.loc[i, "seqID_XY"]
        pos, aa = seqID.split("-")
        aa = aa.split(",")
        index = []
        for p in pos.split(","):
            index.extend(rrm_single.index[rrm_single["position"] == int(p)].tolist())
        seq = wtseq[:index[0]] + aa[0] + wtseq[index[0]+1:index[1]] + aa[1] + wtseq[index[1]+1:]
        rrm[seq] = rrm_double.loc[i, "XY_Enrichment_score"]
#        node += 1
#    sequences = {rrm.nodes[node]["sequence"]: node for node in rrm.nodes}
    # For each sequence, find all its single mutation neighbours and add connections.
    # The max is 75 * 19 = 1425
#    for seq in rrm.nodes:
#        for i in range(len(wtseq)):
#            for aa in amino_acids:
#                if not aa == seq[i]:
#                    fseq = seq[:i] + aa + seq[i+1:]
#                    if fseq in rrm.nodes:
#                        if not (seq, fseq) in rrm.edges(seq):
#                            rrm.add_edge(seq, fseq)
    # NB The RRM dataset contains information about mutations into stop codons "*"
    # This may be useful as a way to investigate the fitness consequences of
    # premature termination, but it is not something I want to consider right away
    # In any case, fitness tends to be low but not zero!
    remove = [seq for seq in rrm.keys() if "*" in seq]
    for seq in remove:
        del rrm[seq]
    return rrm

def build_tem1_landscape():
    tem1_info = pd.read_csv("../data/msu081_Supplementary_Data/Data_S1.csv",
        delimiter=";", decimal=",")
    fitness = {}
    # make the "wild-type sequence" this is not technically the tem1 sequence as the
    # Ambler positions do not necessarily map to codon position in the gene
    wtseq = ''
    positions = {}
    for i,a in enumerate(tem1_info["Ambler Position"].unique()):
        codon = tem1_info.loc[tem1_info["Ambler Position"] == a, "WT codon"].unique()[0]
        wtseq = wtseq + codon
        positions[a] = i * 3
    fitness[wtseq] = 1.0
    for i in range(tem1_info.shape[0]):
        a = tem1_info.loc[i, "Ambler Position"]
        codon = tem1_info.loc[i, "Mutant codon"]
        fit = tem1_info.loc[i, "Fitness"]
        pos = positions[a]
        seq = wtseq[:pos] + codon + wtseq[pos+3:]
        if not np.isnan(fit):
            fitness[seq] = fit
    return fitness


def build_gb1_landscape():
    gb1_info = pd.read_csv("../data/wu_2016_gb1.csv")

    # make graph
    gb1 = nx.Graph()
    # Add edges
    for i in range(gb1_info.shape[0]):
        gb1.add_nodes_from([(gb1_info.loc[i, "Variants"], {
                                "fitness" : gb1_info.loc[i, "Fitness"]})])
    # Add edges
#    sequences = {gb1.nodes[node]["sequence"] : node for node in gb1.nodes}
    missing = set([])
    for seq in gb1.nodes:
        for i in range(len(seq)):
            for aa in amino_acids:
                if not aa == seq[i]:
                    fseq = seq[:i] + aa + seq[i+1:]
                    if fseq in gb1.nodes:
                        if not (seq, fseq) in gb1.edges(seq):
                            gb1.add_edge(seq, fseq)
                    else:
                        missing.add(fseq)
    print("Number of missing sequences: {}".format(len(missing)))
    return gb1

def build_gfp_landscape():
    with open("../data/avGFP_reference_sequence.fa", "r") as f:
        wtntseq = f.readlines()[1] # This sequence also includes the stop codon
    wtseq = translate(wtntseq[:-3], transltable)
    gfpdata = pd.read_table("../data/amino_acid_genotypes_to_brightness.tsv")
    # Make fitness dictionary
    gfp = {}
    # Add the wild-type brightness
    gfp[wtseq] = gfpdata.loc[0,"medianBrightness"]
    for row in range(1,gfpdata.shape[0]):
        mut = gfpdata.loc[row, "aaMutations"]
        muts = mut.split(":")
        mtseq = wtseq
        for m in muts:
            wtaa = m[1]
            pos = int(m[2:-1])
            mtaa = m[-1]
            # Raise error if starting aa is not that in the wild-type.
            # Something odd is going on
            if not wtaa == wtseq[pos]:
                raise RuntimeError("Amino acid at position {} does not match".format(pos))
            # Progressively change mtseq to put in all mutations
            mtseq = mtseq[:pos] + mtaa + mtseq[pos+1:]
        if not "*" in mtseq: # Although those also show brightness!
            gfp[mtseq] = gfpdata.loc[row,"medianBrightness"]
    # add edges
#    for (seq1, seq2) in combinations(gfp.nodes, 2):
#        if hamming(seq1, seq2) == 1:
#            gfp.add_edge(seq1,seq2)
#    for seq in gfp.nodes:
#        for i in range(len(wtseq)):
#            for aa in amino_acids:
#                if not aa == seq[i]:
#                    fseq = seq[:i] + aa + seq[i+1:]
#                    if fseq in gfp.nodes:
#                        if not (seq, fseq) in gfp.edges(seq):
#                            gfp.add_edge(seq, fseq)
    return gfp

def build_his3_landscape():
    s1 = pd.read_table("../data/pokusaeva_2019/S01_fitness.txt")
    his3 = {}
    for row in range(s1.shape[0]):
        seq = s1.loc[row, "AAseq"]
        if not "_" in seq:
            his3[seq] = s1.loc[row, "s"]
#    for seq in his3.nodes:
#        for i in range(len(seq)):
#            for aa in amino_acids:
#                if not aa == seq[i]:
#                    fseq = seq[:i] + aa + seq[i+1:]
#                    if fseq in his3.nodes:
#                        if not (seq, fseq) in his3.edges(seq):
#                            his3.add_edge(seq, fseq)
    # remove sequences that are longer/shorter than most other sequences
    most_common = Counter([len(k) for k in his3.keys()]).most_common(1)[0][0]
    his3 = {k: v for k, v in his3.items() if len(k) == most_common}
    return his3

def build_pard3_landscape():
    # This landscape includes some values that are negative. Shift all data to
    # higher fitnesses
    s1 = pd.read_table("../data/GSE153897_Variant_fitness.csv", delimiter=",")
    pard3_e2 = {}
    pard3_e3 = {}
    e2_negative = 0
    e3_negative = 0
    for row in range(s1.shape[0]):
        seq = s1.loc[row, "Variant"]
        e2 = s1.loc[row, "W_ParE2"]
        e3 = s1.loc[row, "W_ParE3"]
#        pard3_e2[seq] = e2 - s1["W_ParE2"].min()
#        pard3_e3[seq] = e3 - s1["W_ParE3"].min()
        if e2 < 0:
            e2_negative += 1
            pard3_e2[seq] = 0
        else:
            pard3_e2[seq] = e2
        if e3 < 0:
            e3_negative += 1
            pard3_e3[seq] = 0
        else:
            pard3_e3[seq] = e3
    print("Num. ParE2 negative: {}".format(e2_negative))
    print("Num. ParE3 negative: {}".format(e3_negative))
    return pard3_e2, pard3_e3

def check_coverage(landscape, polypeptide=False):
    if not polypeptide:
        # Make sure all are unique
        aaseqs = set([translate(s) for s in landscape.keys()])
    else:
        aaseqs = set([s for s in landscape.keys()])
    # record the coverage for each position in each sequence
    perc_coverage = np.zeros((len(aaseqs), len(list(aaseqs)[0])))
    index = {}
    for i,aasq in tqdm(enumerate(aaseqs)):
        index[aasq] = i
        for j in range(len(aasq)):
            for aa in amino_acids:
                newseq = aasq[:j] + aa + aasq[j+1:]
                if newseq in aaseqs:
                    perc_coverage[i,j] += 5
    return perc_coverage, index

def get_median_completeness(perc_coverage):
    # How complete is the neighbourhood of the median sequence?
    return np.median(np.sum(perc_coverage, axis=1)/(perc_coverage.shape[1]*100))*100


def read_amino_acids():
    codontable = read_codon_table()
    amino_acids = list(codontable.keys())
    amino_acids.remove("*")
    return amino_acids

amino_acids = read_amino_acids()

def write_fitness_dictionary(fname, fitdict):
    with open(fname, "w") as f:
        for k,v in fitdict.items():
            line = k + "," + str(v)+"\n"
            f.write(line)


if __name__ == '__main__':
    pass

#rrm.number_of_edges()
#rrm.number_of_nodes()
# See degree distribution
#plt.hist([np.log10(rrm.degree[n]) for n in rrm.nodes]); plt.show()
# See fitness distribution
#plt.hist([n["fitness"] for n in rrm.nodes if n["fitness"] !== np.nan]); plt.show()
#how many nans?
#np.sum([n["fitness"] == np.nan for n in rrm.nodes])

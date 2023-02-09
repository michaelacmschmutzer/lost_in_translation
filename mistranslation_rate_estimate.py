"""Curate Mordret et al (2019) data and extract mistranslation rates"""

# The paper claims (p. 428) that the proteomics data is derived from MG1655 WT
# in MOPS Complete medium. This combination does not exist in Table S1.
# Assume an error. Use BW25113 (WT) with MOPS Complete medium, which has 17990
# datapoints

# A few samples had an intensity ratio greater than one. Ignore these. Or look how many datapoints contributed to these

import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt

def read_amino_acids():
    codontable = read_codon_table()
    amino_acids = list(codontable.keys())
    amino_acids.remove("*")
    return amino_acids

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

def get_rates():
    ecolimr = pd.read_csv("../data/MordretTableS1.csv")

    index = (ecolimr["Strain"]=="BW25113 (WT)")&(ecolimr["Media"]=="MOPS Complete")
    #index = (ecolimr["Strain"]=="MG1655 WT")&(ecolimr["Media"]=="LB")
    #index = ecolimr["Media"]=="MOPS Complete"
    subset = ecolimr[index&(ecolimr["Timepoint"]=="2")]
    codons = list(subset["Origin codon"].unique())

    # Assume that a certain transition does not occur if it is not observed (rate = 0)
    misrates = {a+b+c : {aa : [0,0,0] for aa in amino_acids}
                for a,b,c in product(nucleotides,repeat=3)
                if not a+b+c in stopcodons}
    for c in codons:
        aa = subset.loc[subset["Origin codon"]==c,"Destination AA"].to_list()
        ir = subset.loc[subset["Origin codon"]==c,"Intensities ratio"].to_list()
        bp = subset.loc[subset["Origin codon"]==c,"Base peptide intensity"].to_list()
        dp = subset.loc[subset["Origin codon"]==c,"Dependent peptide intensity"].to_list()
        rates = {}
        for i,a in enumerate(aa):
            if a in rates.keys():
                # Figure 2B - only use datapoints for those where both bp and dp were
                # measured (I take this to mean that dp and bp are both > 0)
                if bp[i] > 0 and dp[i] > 0:
                    rates[a].append(ir[i])
            else:
                if bp[i] > 0 and dp[i] > 0:
                    rates[a] = [ir[i]]
        for a,r in rates.items():
            medr = np.nanmedian(r)
            misrates[c][a] = [medr, np.nanvar(r), len(r)]
        # remove entry for cognate amino acid
        caa = subset.loc[subset["Origin codon"]==c,"Origin AA"].unique()
        if len(caa) > 0:
            caa = caa[0]
            del misrates[c][caa]
    return misrates

def get_summary(misrates):
    raw=[]
    for c,v in misrates.items():
        for aa, r in v.items():
            if not r[1] == 0:
                raw.append(r)
    nobs = np.array([r[2] for r in raw])
    err = np.array([ 1.96*np.sqrt(r[1]) / np.sqrt(r[2]) for r in raw])
    irmed = np.array([r[0] for r in raw])
    return nobs, irmed, err

def regression(nobs, irmed, nobs_threshold=18):
    index = nobs >= nobs_threshold
    res = stats.linregress(nobs[index], irmed[index])
    return res

def plot_misrates(misrates, nobs_threshold=18):
    nobs, irmed, err = get_summary(misrates)
    # simple linear regression
    res = regression(nobs, np.log10(irmed), nobs_threshold=0)
    respart = regression(nobs, np.log10(irmed), nobs_threshold=nobs_threshold)
    fig = plt.figure(figsize=(4.8,3.6))
    plt.plot(nobs, irmed, '+', color="black")
    plt.plot(np.array(range(max(nobs)+1)), 10**(res.intercept + res.slope*np.array(range(max(nobs)+1))), 'r', label="no threshold")
    plt.plot(np.array(range(max(nobs)+1)), 10**(respart.intercept + respart.slope*np.array(range(max(nobs)+1))), 'blue', label="with threshold")
    plt.errorbar(nobs, irmed, yerr=err, linestyle="", color="grey")
    plt.yscale("log")
    plt.xlabel("Number of samples", fontsize=14)
    plt.ylabel("Mistranslation rate estimate", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14, frameon=False)
    plt.tight_layout(h_pad=0.1)
    plt.show()

def find_minimum_translation_rate(misrates):
    nobs, irmed, err = get_summary(misrates)
    minrates = []
    excluded = []
    for i in range(51):
        res = regression(nobs, np.log10(irmed), i)
        minrates.append(10**res.intercept)
        excluded.append(sum(nobs < i)/len(nobs)*100)
    fig, ax = plt.subplots(figsize=(5,3.6))
    color = "black"
    ax.plot(range(51), np.log10(minrates), color="black")
    ax.set_xlabel("Threshold for exclusion\n(minimum number of samples)", size=14)
    ax.set_ylabel("Minimum mistranslation rate" "\n" r"log$_{10}$(regression intercept)", color=color, size=14)
    ax.tick_params(axis='y', labelcolor=color, labelsize=14)
    ax.tick_params(axis='x', labelcolor=color, labelsize=14)
    ax.set_xlim(0,50)
    ax2 = ax.twinx()
    color = 'tab:red'
    ax2.set_ylabel("Percentage of data excluded", color=color, size=14)  # we already handled the x-label with ax1
    ax2.plot(range(51), excluded, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
    ax2.set_ylim(0,100)
    fig.tight_layout()
    fig.show()
    return np.array([[a,b] for a,b in zip(range(51),minrates)])

def interpolate_mistranslation_rates(misrates, threshold=18):
    # Only replace those with less than threshold observations
    nobs, irmed, err = get_summary(misrates)
    res = regression(nobs, irmed, threshold)
    for c,aas in misrates.items():
        for aa,r in aas.items():
            if r[2] < threshold: # Number of observations less than threshold
                misrates[c][aa][0] = 10**(r[2] * res.slope + res.intercept)
    return misrates

def save_rates(misrates, fname="../data/misrates.csv"):
    with open(fname, "w") as f:
        for c,aas in misrates.items():
            for aa,r in aas.items():
                f.write(",".join([c,aa]+[str(i) for i in r])+"\n")



# Many of the intensity ratios for rarely observed substitutions are very high,
# and some exceed one. These are probably artifacts.

#for a,b,c in product(["A","T","C","G"], repeat=3):
#    if not a+b+c in codons:
#        print(a+b+c)

def cleanup(misrates, newval = 1e-5):
    for c,aa in misrates.items():
        for a,r in aa.items():
            if r[0] > 1 or r[2] < 10:
                misrates[c][a][0] = newval
    return misrates

def mistranslationrates():
    misrates = get_rates()
    misrates = cleanup(misrates)
    return misrates

amino_acids = read_amino_acids()
nucleotides = ["A", "T", "G", "C"]
stopcodons = ["TAA", "TAG", "TGA"]

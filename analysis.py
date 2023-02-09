#!/usr/bin/env python3

"""Analysis and plotting"""

import os
import itertools
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import evolution as evo
import sqlite3 as sql
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mplcolors
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from collections import Counter, OrderedDict
from scipy import stats, interpolate
from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess
from  scikit_posthocs import posthoc_nemenyi_friedman as nemenyi
from  scikit_posthocs import posthoc_dunn as dunn
from neutral_networks import find_neutral_neighbours
import pickle

import palettable as pt
cork4 = pt.scientific.diverging.Cork_4.mpl_colormap

weszissou = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"]
weszissou.reverse()
wes = sns.blend_palette(weszissou, as_cmap=True)
# colors for the three fitness landscapes (use 3 out of 5)
zissou_3 = pt.wesanderson.Zissou_5.hex_colors[1:4]
# blended palettes: single colours
redscale = sns.blend_palette(["grey", wes(0.)], as_cmap=True)
whitered = sns.blend_palette(["white", wes(0.)], as_cmap=True)
whitegreen = sns.blend_palette(["white", '#046307'], as_cmap=True)
greenbrown = sns.blend_palette(['#046307', '#633304'], as_cmap=True)
whiteblue = sns.blend_palette(["white", wes(0.9999)], as_cmap=True)
bluescale = sns.blend_palette(["grey", wes(0.9999)], as_cmap=True)
darkbluescale = sns.blend_palette(["white", zissou_3[0]], as_cmap=True)




################################################################################
#                  S E L E C T I O N    P O W E R                              #
################################################################################


def make_selection_power_csv(directory):
    # Read in results, build a dataframe from them, and save it as a csv
    # Then remove the txt files
    fnames = [f for f in os.listdir(directory) if f.startswith("results_") and f.endswith(".txt")]
    results = []
    for f in fnames:
        with open(directory+f, "r") as fi:
            line = fi.read()
            entries = line.split(",")
            # The first two entries are ntseqs, the rest numeric
            results.append(entries[:2]+[float(e) for e in entries[2:]])
    results = pd.DataFrame(results, columns=["wt","mt","fit_wt", "fit_mt",
                "fit_wt_mis", "fit_wt_var", "fit_mt_mis", "fit_mt_var",
                "scoeff", "scoeff_mis", "N", "Ne", "fixprob", "fixprob_mis",
                "prnum", "psum_wt", "psum_mt"])
    return results

def plot_figure_1A(results, landscape, save_figure="./fitness_change.pdf"):
    # from https://stackoverflow.com/questions/37008112/matplotlib-plotting-histogram-plot-just-above-scatter-plot
    fitch = results["fit_wt_mis"].values - results["fit_wt"].values
    fig = plt.figure(figsize=(4.8,4.8), constrained_layout=False)
    gs = gridspec.GridSpec(6, 6)
    ax_main = fig.add_subplot(gs[1:, :5])
    ax_xDist = fig.add_subplot(gs[0, :5],sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:, 5:],sharey=ax_main)
    ax_main.scatter(results["fit_wt"], fitch, marker='o', linewidths=0, color="grey", alpha=0.5)
    ax_main.set_ylabel("Change in fitness", fontsize=16)
    ax_main.set_xlabel("Fitness (no mistranslation)",fontsize=16)
    ax_main.tick_params(axis='both', labelsize=14)
    ax_main.axhline(0, color=redscale(0.9999), linestyle="--")
    ax_main.axvline(np.mean([v for v in landscape.values()]), color=redscale(0.9999), linestyle="--")
    ax_xDist.hist(results["fit_wt"],bins=100,align='mid', color="grey")
    ax_xDist.tick_params(axis='both', labelsize=14)
    ax_xDist.set_ylabel("Count", fontsize=16)
    ax_xDist.set_yscale("log")
    ax_yDist.hist(fitch,bins=100,orientation='horizontal',align='mid', color="grey")
    ax_yDist.tick_params(axis='both', labelsize=14)
    ax_yDist.set_xlabel('Count', fontsize=16)
    ax_yDist.set_xscale("log")
    ax_yDist.set_xticks([100])
    plt.setp(ax_xDist.get_xticklabels(), visible=False)
    plt.setp(ax_yDist.get_yticklabels(), visible=False)
    plt.tight_layout()
    if save_figure:
        fig.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def plot_fitness_effective_population_size(results):
    plt.plot(results["fit_wt"], results["Ne"], ".", mec='none', color="grey", alpha=0.4)
    plt.xlabel("Fitness (no mistranslation)")
    plt.ylabel("Effective population size")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

def plot_figure_1C(landscape="gb1", exprlevel = [1,10,100,500,1000]):
#    mean_ne = []
#    quan25 = []
#    quan75 = []
#    stderr_ne = []
    ne = []
    for i in exprlevel:
        results = pd.read_csv("../results/selection_power/"+landscape+"/proteins{}_popsize1e6/results.csv".format(i))
        for v in results["Ne"].to_list():
            ne.append([i, v])
        #mean_ne.append(results["Ne"].mean())
        #quan25.append(mean_ne[-1] - np.quantile(results["Ne"], 0.025))
        #quan75.append(np.quantile(results["Ne"], 0.975) - mean_ne[-1])
#        stderr_ne.append(results["Ne"].std())#/np.sqrt(results["Ne"].mean()))
    ne = pd.DataFrame(ne, columns=["pr", "ne"])
    fig, ax = plt.subplots(figsize=(4.8,3.6))
    sns.violinplot(x="pr", y="ne", data=ne, color="grey", scale="width")
#    ax.plot(exprlevel, mean_ne, '-o', color="grey")
#    plt.plot(exprlevel, mean_ne, '-o', color="grey")
#    plt.errorbar(exprlevel, mean_ne, yerr=[quan25,quan75], color="grey", linestyle="--", capsize=2)
    ax.set_ylim(bottom=-5e3,top=1e6*1.01)
    ax.set_xlabel("Protein expression level (per cell)", fontsize=16)
    ax.set_ylabel(r"Effective population size ($N_e$)", fontsize=16)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y", useMathText=True, useOffset=False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
#    ax.set_yscale("log")
#    ax.set_xscale("log")
    plt.tight_layout()
    plt.show()

def plot_effective_popsize(results, save_figure="./effective_popsize.pdf"):
    plt.hist(results["Ne"], bins=100, color="black")
    plt.yscale("log")
    plt.xlabel("Effective population size (mistranslation)")
    plt.ylabel("Number of observations")
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def plot_fixprob_hist(results, save_figure="./fixprobhist.pdf"):
    plt.hist([results["fixprob"], results["fixprob_mis"]], bins=50,
        color=["black", "red"], label=["without", "with"])
    plt.yscale("log")
    plt.xlabel("Fixation probability")
    plt.ylabel("Number of observations")
    plt.legend()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def plot_barplot_fitness_change(landscapes = ["GB1", "ParD3E2", "ParD3E3"], Ne=1e6, pr=1):
    fitchange = []
    for ldsc in landscapes:
        results = pd.read_csv("../results/selection_power/{}/".format(ldsc.lower()) +
            "proteins{}_popsize1e{}/results.csv".format(pr, int(np.log10(Ne))))
        fch = results["fit_wt_mis"].values - results["fit_wt"].values
        for ch in fch:
            fitchange.append([ldsc, ch])
    fitchange = pd.DataFrame(fitchange, columns=["landscape", "Change in fitness"])
    ax = sns.boxplot(x="landscape", y="Change in fitness", data=fitchange)
    ax.set_xlabel("")
    plt.show()

def plot_figure_1G(results, save_figure=None):
    if not "class" in results.columns:
        results = classify_results(results)
    N = results.loc[0, "N"]
    benidx = (results["class"] == results["class_mis"]) & (results["class"] == "beneficial")
    bendiff = results.loc[benidx, "fixprob_mis"] - results.loc[benidx, "fixprob"]
    fig = plt.figure(figsize=(4.8,3.6))
    plt.hist(bendiff, bins=100, color=wes(0.99), align="mid")
    plt.yscale("log")
    plt.ylabel("Frequency", fontsize=16)
    plt.xlabel("Change in fixation probability\nof beneficial mutations", fontsize=16)
    plt.locator_params(axis='x', nbins=6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if save_figure:
        plt.savefig(save_figure, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    plt.show()

def plot_figure_1H(Ne=1e6, pr=1, save_figure=None):
    num_increase = []
    landscapes = ["GB1", "ParD3E2", "ParD3E3"]
    landscape_labels = ["Antibody-\nbinding", "Toxin-\nantitoxin\n(E2)", "Toxin-\nantitoxin\n(E3)"]
    for ldsc in landscapes:
        results = pd.read_csv("../results/selection_power/{}/".format(ldsc.lower()) +
            "proteins{}_popsize1e{}/results.csv".format(pr, int(np.log10(Ne))))
        results = classify_results(results)
        benidx = (results["class"] == results["class_mis"]) & (results["class"] == "beneficial")
        bendiff = results.loc[benidx, "fixprob_mis"] - results.loc[benidx, "fixprob"]
        num_increase.append(sum(bendiff > 0)/len(bendiff)*100)
    print(num_increase)
    fig = plt.figure(figsize=(4.8,3.6))
    plt.bar(landscapes, num_increase, color="grey")
    plt.ylabel("% beneficial mutations with \n increased fixation probability", fontsize=16)
    plt.xticks(ticks=[0,1,2], labels=landscape_labels, fontsize=16)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust()
    if save_figure:
        plt.savefig(save_figure, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_relative_fixation_prob(results):
    N = results.loc[0, "N"]
#    index = (results["class"] == results["class_mis"]) & (results["class"] != "neutral")
    benidx = (results["class"] == results["class_mis"]) & (results["class"] == "beneficial")
    delidx = (results["class"] == results["class_mis"]) & (results["class"] == "deleterious")
#    diff = results.loc[index, "fixprob_mis"] - results.loc[index, "fixprob"]
    bendiff = results.loc[benidx, "fixprob_mis"] - results.loc[benidx, "fixprob"]
    deldiff = results.loc[delidx, "fixprob_mis"] - results.loc[delidx, "fixprob"]
    fig, ax = plt.subplots(1,2, sharey=True)#, sharex=True, sharey=True)
#    ax[0].hist(diff, bins=100, color="black", align="mid")
#    ax[0].set_title("all non-neutral")
#    ax[0].set_yscale("log")
#    ax[0].set_ylabel("Number of occurances")
    ax[0].hist(bendiff, bins=100, color="black", align="mid")
    ax[0].set_yscale("log")
    ax[0].set_title("beneficial")
    ax[1].ticklabel_format(axis="x", useMathText=True, useOffset=True)
    ax[1].hist(deldiff, color="black", align="mid")
    ax[1].set_title("deleterious")
    ax[1].set_yscale("log")
    ax[1].ticklabel_format(axis="x", useMathText=True)
    fig.text(0.5, 0.03, "Change in fixation probability", ha='center')
    plt.show()

def plot_change_fixprob(results, save_figure="./changefixprob.pdf"):
    # Check that results has fitness classification, else do it
    def myticks(x,pos):
        if x == 0:
            return "$0$"
        sign = np.sign(x)
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = sign * abs(x)/10**exponent

        return r"${:2.0f} \times 10^{{ {} }}$".format(coeff,exponent)

    if not "class" in results.columns:
        results = classify_results(results)
    # Filter out those that are neutral or change classification
    index = (results["class"] == results["class_mis"]) & (results["class"] != "neutral")
    benidx = (results["class"] == results["class_mis"]) & (results["class"] == "beneficial")
    delidx = (results["class"] == results["class_mis"]) & (results["class"] == "deleterious")
    diff = results.loc[index, "fixprob_mis"] - results.loc[index, "fixprob"]
    bendiff = results.loc[benidx, "fixprob_mis"] - results.loc[benidx, "fixprob"]
    deldiff = results.loc[delidx, "fixprob_mis"] - results.loc[delidx, "fixprob"]
    fig, ax = plt.subplots(1,3, sharey=True)#, sharex=True, sharey=True)
    ax[0].hist(diff, bins=100, color="black", align="mid")
    ax[0].set_title("all non-neutral")
    ax[0].set_yscale("log")
    #ax[0].set_xlim(-1, 1)
    ax[0].set_ylabel("Number of occurances")
    ax[1].hist(bendiff, bins=100, color="black", align="mid")
    #ax[1].set_xlim(-1, 1)
    ax[1].set_yscale("log")
    ax[1].set_title("beneficial")
    ax[1].set_xlabel("Change in fixation probability")
    ax[2].hist(deldiff, color="black", align="mid")
    ax[2].set_title("deleterious")
    ax[2].set_yscale("log")
    ax[2].xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
#    plt.yscale("log")
#    plt.xlim(-1, 1)
#    plt.xlabel("Change in fixation probability")
#    plt.ylabel("Number of observations")
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def direction_fitness_change(results):
    # H: The majority of fitness changes due to mistranslation are a decrease in
    # fitness
    if len(results["N"].unique()) == 1:
        N = results["N"].unique()[0]
    else:
        raise RuntimeError("More than one value of N in dataframe")
    fitdiff = list(results["fit_mt_mis"] - results["fit_mt"])
    fitdiff.extend(list(results["fit_wt_mis"] - results["fit_wt"]))
    fitdiff = np.array(fitdiff)
    plt.hist(fitdiff, bins=100, color="black")
    plt.yscale("log")
    plt.xlabel("Fitness difference")
    plt.ylabel("Number of observations")
    plt.show()
    effneutral = fitdiff[(fitdiff * N < 0.25) & (fitdiff * N > -0.25)]
    len(effneutral) / len(fitdiff) # percent of effectively neutral changes

def change_fixation_prob(results):
    # Get rid of wt fitness == 0 ??
    index = results["fit_wt"] != 0
    plotdf = results.loc[index,["fixprob", "fixprob_mis"]].melt(var_name="mis", value_name="fixprob")
    ax = sns.violinplot(x="mis", y="fixprob", data=plotdf)
    ax.set_xticklabels(["no mistranslation", "mistranslation"])
    ax.set_ylim(0,1)
    ax.set_xlabel("")
    ax.set_ylabel("Fixation probability")
    plt.show()


def prepare_figure_1B(results, gb1):
    ntseqs = results["wt"]
    nresults = []
    for nt in tqdm(ntseqs):
        # fitness without mistranslation
        aa = evo.translate(nt)
        fit = gb1[aa]
        # Get all nucleotide sequences
    #    ntseqs = evo.reverse_translate(aa)
        # Get mistranslation fitnesses where available
        mfit = results.loc[results["wt"] == nt, "fit_wt_mis"].values
        # Get fitnesses of genetic neighbours without mistranslation
        neigh = evo.find_neighbours(nt)
        nfit = np.mean([gb1[evo.translate(ns)] for ns in neigh if evo.translate(ns) in gb1.keys()])
        # Get fitnesses of phenotypic neighbours (1 amino acid step away)
        pheno_neighbours = []
        for i in range(len(aa)):
            for amino in evo.amino_acids:
                naa = aa[:i] + amino + aa[i+1:]
                if naa in gb1.keys(): # ignore...
                    pheno_neighbours.append(gb1[naa])
        pfit = np.mean(pheno_neighbours)
        # get proportion of proteins mistranslation
        misrate = get_prop_mistranslated(nt)
        nresults.append([nt, aa, fit, mfit, nfit, pfit, misrate])
    nresults = pd.DataFrame(nresults, columns=
        ["nt", "aa", "fit_nomis", "fit_mis", "fit_neigh_nt", "fit_neigh_aa", "misrate"])
    return nresults

def plot_figure_1B(nresults, save_figure=None):
    # No mistranslation
    print(stats.kendalltau(nresults["fit_neigh_nt"], nresults["fit_neigh_aa"]))
    fig = plt.figure(figsize=(4.8,3.6))
    plt.plot(nresults["fit_neigh_nt"], nresults["fit_neigh_aa"], ".",
        color="grey", mec='none', alpha=0.4)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean fitness of \n genetic neighbours", fontsize=16)
    plt.ylabel("Mean fitness of \n phenotypic neighbours", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def classify_fitness_effects(scoeff, effpop):
    classify = []
    for s,N in zip(scoeff,effpop):
        if s * N > 0.25:
            classify.append("beneficial")
        elif s * N < -0.25:
            classify.append("deleterious")
        else:
            classify.append("neutral")
    return classify

def classify_results(results):
    if len(results["N"].unique()) == 1:
        N = results["N"].unique()[0]
    else:
        raise RuntimeError("More than one value of N in dataframe")
    scoeff = results["scoeff"].values
    effpop = results["N"].values
    scoeff_mis = results["scoeff_mis"].values
    effpop_mis = results["Ne"].values
    results["class"] = classify_fitness_effects(scoeff, effpop)
    results["class_mis"] = classify_fitness_effects(scoeff_mis, effpop_mis)
    return results

def plot_scoeff_fixation_probability_mistranslation(results):
    plt.scatter(results["scoeff_mis"], results["fixprob_mis"], s=5*np.log10(results["Ne"]))
    plt.xlabel("Selection coefficient")
    plt.ylabel("Probability of fixation")
    plt.xlim(-1.5, 30)
    plt.show()

def make_sankey_diagram_figure_1D(results, save_figure="../results/sankey_diagram.pdf"):
    # Plot change in selection coefficients
    if len(results["N"].unique()) == 1:
        N = results["N"].unique()[0]
    else:
        raise RuntimeError("More than one value of N in dataframe")
    # classify as beneficial, effectively neutral or deleterious
    scoeff = results["scoeff"].values
    effpop = results["N"].values
    scoeff_mis = results["scoeff_mis"].values
    effpop_mis = results["Ne"].values
    # Use N here or the Ne??
    classify = classify_fitness_effects(scoeff, effpop)
    classifymis = classify_fitness_effects(scoeff_mis, effpop_mis)
    trans = np.zeros((9,)) # benebene, beneneut, benedel, neutbene, neutneut, neutdel, delbene, delneut, deldel
    for c,cm in zip(classify,classifymis):
        if c == "beneficial" and cm == "beneficial":
            trans[0] +=1
        elif c == "beneficial" and cm == "neutral":
            trans[1] +=1
        elif c == "beneficial" and cm == "deleterious":
            trans[2] +=1
        elif c == "neutral" and cm == "beneficial":
            trans[3] +=1
        elif c == "neutral" and cm == "neutral":
            trans[4] +=1
        elif c == "neutral" and cm == "deleterious":
            trans[5] +=1
        elif c == "deleterious" and cm == "beneficial":
            trans[6] +=1
        elif c == "deleterious" and cm == "neutral":
            trans[7] +=1
        elif c == "deleterious" and cm == "deleterious":
            trans[8] +=1
        else:
            raise RuntimeError("Unknown classification")

    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 41,
      line = dict(color = "black", width = 0.5),
      label =  [""] * 6, #["beneficial", "neutral", "deleterious", "beneficial_mistranslation",
                #      "neutral_mistranslation", "deleterious_mistranslation"],
      color = [mplcolors.rgb2hex(c) for c in [wes(0.9999), wes(0.5), wes(0.0)] * 2]#["blue", "white", "red", "blue", "white", "red"]
    ),
    link = dict(
      source = [0, 0, 0, 1, 1, 1, 2, 2, 2], # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = [3, 4, 5, 3, 4, 5, 3, 4, 5],
      value = trans
    ))])

    # make space for explanation / annotation
    fig.update_layout(margin=dict(l=60, r=60, t=60, b=60), width=600, height=450)

    # add annotation
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=-0.1,
                                        y=1.12,
                                        showarrow=False,
                                        text="Without mistranslation",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=1.1,
                                        y=1.12,
                                        showarrow=False,
                                        text="With mistranslation",
                                        textangle=0,
                                        xanchor='right',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0.08,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Beneficial",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0.42,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Neutral",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0.72,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Deleterious",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_shape(type="rect",
    x0=0, y0=-0.06, x1=0.07, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.9999)))
    fig.write_image(save_figure)
    fig.add_shape(type="rect",
    x0=0.35, y0=-0.06, x1=0.41, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.5)))
    fig.add_shape(type="rect",
    x0=0.63, y0=-0.06, x1=0.71, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.0)))
    fig.write_image(save_figure)

def plot_figure_1F(directory="../results/selection_power/gb1/", save_figure=None):
    perc_neut = {}
    fig = plt.figure(figsize=(4.8,3.6))
    for c,n in enumerate([4, 6, 8]):
        perc_neut[n] = []
        for p in [1, 10, 100]:
            results = pd.read_csv(directory+"proteins{}_popsize1e{}/results.csv".format(p,n))
            results = classify_results(results)
            perc = sum([v=="neutral" for v in results["class_mis"]]) / results.shape[0]
            perc_neut[n].append(perc * 100)
        plt.plot([1, 10, 100], perc_neut[n], "-o", label = r"10$^{}$".format(n),
            color = redscale(c/2))
    plt.xlabel("Number of proteins per cell", fontsize=16)
    plt.ylabel("Percentage neutral \nwith mistranslation", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    legend = plt.legend(loc="center right", fontsize=14, frameon=False,
        bbox_to_anchor=(1.,0.55))
    legend.set_title('Population size',prop={'size':14})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def plot_si_figure_1F(directory="../results/selection_power/pard3e3/", save_figure=None):
    perc_neut = {}
    fig = plt.figure(figsize=(4.8,3.6))
    for c,n in enumerate([4, 6, 8]):
        perc_neut[n] = []
        for p in [1, 500]:
            results = pd.read_csv(directory+"proteins{}_popsize1e{}/results.csv".format(p,n))
            results = classify_results(results)
            perc = sum([v=="neutral" for v in results["class_mis"]]) / results.shape[0]
            perc_neut[n].append(perc * 100)
        plt.plot([1, 500], perc_neut[n], "-o", label = r"10$^{}$".format(n),
            color = redscale(c/2))
    plt.xlabel("Number of proteins per cell", fontsize=16)
    plt.ylabel("Percentage neutral \nwith mistranslation", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    legend = plt.legend(loc="center right", fontsize=14, frameon=False,
        bbox_to_anchor=(1.,0.55))
    legend.set_title('Population size',prop={'size':14})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def synonymous_mutation_neutrality(results):
    # Do synonymous mutations become non-neutral in the presence of mistranslation?
    if len(results["N"].unique()) == 1:
        N = results["N"].unique()[0]
    else:
        raise RuntimeError("More than one value of N in dataframe")
    scoeff = results["scoeff"].values
    popsize = results["N"].values
    scoeff_mis = results["scoeff_mis"].values
    popsize_mis = results["Ne"].values
    # Use N here or the Ne??
#    seffnt = scoeff[(scoeff * N < 0.25) & (scoeff * N > -0.25)]
#    smiseffnt = scoeff_mis[(scoeff_mis * N < 0.25) & (scoeff_mis * N > -0.25)]
    results["class"] = classify_fitness_effects(scoeff, popsize)
    results["class_mis"] = classify_fitness_effects(scoeff_mis, popsize_mis)
#    synfates = []
#    for i in range(results.shape[0]):
#        wtseq = results.loc[i, "wt"]
#        mtseq = results.loc[i, "mt"]
#        wtaa = evo.translate(wtseq, evo.transltable)
#        mtaa = evo.translate(mtseq, evo.transltable)
#        if wtaa == mtaa: # synonymous
#            synfates.append([results.loc[i,"class"], results.loc[i, "class_mis"]])
    # Other way round... starting with neutral
    neutral = []
    for i in range(results.shape[0]):
        if results.loc[i, "class"] == "neutral":
            wtseq = results.loc[i, "wt"]
            mtseq = results.loc[i, "mt"]
            wtaa = evo.translate(wtseq, evo.transltable)
            mtaa = evo.translate(mtseq, evo.transltable)
            if wtaa == mtaa: # synonymous
                neutral.append([True, results.loc[i,"class"], results.loc[i, "class_mis"]])
            else:
                neutral.append([False, results.loc[i,"class"], results.loc[i, "class_mis"]])
    return pd.DataFrame(neutral, columns=["syn", "class", "class_mis"])

def change_selection_strength(results):
    # Easier: to see difference in selection strengths/coefficients
    # selstrength.loc[selstrength["class"]=="beneficial"].mean()
    # Also consider fixation probabilities!
    if len(results["N"].unique()) == 1:
        N = results["N"].unique()[0]
    else:
        raise RuntimeError("More than one value of N in dataframe")
    if not "class" in results.columns:
        results = classify_results(results)
    # Avoid mutations that change classification or that are neutral
    slcstr = []
    for c in ["beneficial", "deleterious"]:
        index = (results["class"] == c) & (results["class_mis"] == c)
#        ratio = (results.loc[index, "scoeff_mis"] / results.loc[index, "scoeff"]).to_list()
#        slcstr.extend([[np.log10(r), c] for r in ratio])
        change = (results.loc[index, "fixprob_mis"] - results.loc[index, "fixprob"])
        slcstr.extend([ [ch, c] for ch in change])
    selstrength = pd.DataFrame(slcstr, columns=["changefixprob", "class"])
    sns.boxplot(selstrength.loc[selstrength["class"]=="beneficial", "changefixprob"])
    plt.show()
    sns.boxplot(selstrength.loc[selstrength["class"]=="deleterious", "changefixprob"])
    plt.show()
    return selstrength


def plot_correlation_with_phenotypic_neighbours(nresults):
    # No mistranslation
    plt.plot(nresults["fit_nomis"], nresults["fit_neigh_aa"], "k+")
#    plt.xscale("log")
#    plt.yscale("log")
    plt.xlabel("Fitness (no mistranslation)")
    plt.ylabel("Mean fitness of phenotypic neighbours")
    plt.show()

def plot_figure_1E(results, save_figure=None):
    syn = []
    non = []
    data = []
    for row in range(results.shape[0]):
        wt = results.loc[row, "wt"]
        mt = results.loc[row, "mt"]
        # Only measure differences with mistranslation
        diff = abs(results.loc[row, "fit_wt_mis"] -results.loc[row, "fit_mt_mis"])
        if diff != 0:
            if evo.hamming(evo.translate(wt), evo.translate(mt)) == 0:
                data.append(["synonymous", diff])
            else:
                data.append(["nonsynonymous", diff])
    data = pd.DataFrame(data, columns=["kind", "difference"])
    fig = plt.figure(figsize=(4.8,3.6))
    ax = sns.histplot(x="difference", hue="kind", data=data, multiple="dodge",
                log_scale=10, palette=[mplcolors.rgb2hex(greenbrown(0.0)),
                mplcolors.rgb2hex(greenbrown(0.9999))], alpha=1,
                edgecolor='none')
#    plt.hist(syn, bins=slogbins, label="synonymous")
    ax.set_ylabel("Number of observations", fontsize=16)
    ax.set_xlabel("Absolute difference in fitness", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.get_legend().set_title(None)
    plt.setp(ax.get_legend().get_texts(), fontsize=14)
    plt.tight_layout()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def investigate_deviation_from_mean_aa_fit(results):
    devi = []
    misrates = []
    fitch = []
    aaneighs = []
    corrs = []
    for row in range(aameans.shape[0]):
        seq = aameans.loc[row, "aa"]
        mfit = aameans.loc[row, "fit_mis"]
        ds = results.loc[results["aa"] == seq, "fit_mis"].values - mfit
        mr = results.loc[results["aa"] == seq, "misrate"].values
        fc = results.loc[results["aa"] == seq, "fit_mis"].values - results.loc[results["aa"] == seq, "fit_nomis"].values
        an = results.loc[results["aa"] == seq, "fit_neigh_aa"].values
        if len(mr) > 1:
            corrs.append(stats.spearmanr(mr, ds).correlation)
        for m,d,f,a in zip(mr,ds,fc, an):
            devi.append(d)
            misrates.append(m)
            fitch.append(f)
            aaneighs.append(a)
    plt.plot(aameans["fit_mis"], corrs, "k+")
    plt.xscale("log")
    plt.show()


################################################################################
#                N E U T R A L    N E T W O R K S                              #
################################################################################

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

def make_nearly_neutral_network_csv(directory="../results/neutral_networks/"):
    nearly_neutral = []
    for ldsc in ["GB1", "ParD3E2", "ParD3E3"]:
        flist = os.listdir(directory+ldsc.lower()+"/")
        for f in flist:
            if f.endswith("pkl"):
                params = f.split(".")[0]
                params = params.split("_")
                if params[0] == "no":
                    pr = "Nan"
                    N = r"10$^{"+params[2][-1]+"}$"
                else:
                    pr = params[1]
                    N = r"10$^{"+params[2][-1]+"}$"
                if len(params) == 4:
                    kind = params[3]
                elif pr == "Nan":
                    kind = "no mistranslation"
                else:
                    kind = "both"
                with open(directory+ldsc.lower()+"/"+f, "rb") as fi:
                    networks = pickle.load(fi)
                neutral = restructure_networks(networks)
                sizes = get_network_sizes(neutral)
                for s in sizes:
                    row = [ldsc, pr, N, kind, s]
                    nearly_neutral.append(row)
    return pd.DataFrame(nearly_neutral, columns=["landscape", "pr_num", "N", "kind", "size"])

def number_of_networks(nearly_neutral):
    count = []
    for ldsc in ["GB1", "ParD3E2", "ParD3E3"]:
        for pr in ["Nan", "1", "500"]:
            condition = ( (nearly_neutral["landscape"] == ldsc) &
                        (nearly_neutral["pr_num"] == pr) &
                        ( (nearly_neutral["kind"]=="no mistranslation") |
                        (nearly_neutral["kind"]=="both") ) &
                        (nearly_neutral["N"] == r"10$^{4}$") )
            count.append([ldsc, pr, sum(condition)])
    return count

def plot_nearly_neutral_networks_all(nearly_neutral, pr=1, kind="both"):
    landscape_labels = {"GB1": "Antibody-\nbinding",
                        "ParD3E2": "Toxin-\nantitoxin\n(E2)",
                        "ParD3E3": "Toxin-\nantitoxin\n(E3)"}
    # Use only the observations with the full effect of mistranslation
    condition = (nearly_neutral["kind"]==kind) & (nearly_neutral["pr_num"] == str(pr))
    subset = nearly_neutral.loc[condition]
    subset = subset.reset_index()
#    subset.loc[:,"logSize"] = np.log10(subset.loc[:,"size"])
    order = [r"10$^{4}$", r"10$^{6}$", r"10$^{8}$"]
    fig = plt.figure(figsize=(4.8,4.8))
    ax = sns.violinplot(x="landscape", y="size", hue="N", data=subset,
                hue_order=order, palette=[mplcolors.rgb2hex(redscale(v))
                for v in (0.0, 0.5, 1.0)],
                saturation=1,
                showmeans=True,
                scale="width",
                meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"})
    ax.set_ylabel(r"Network size (log$_{10}$)", fontsize=16)
    ax.set_xlabel("")
    plt.legend(bbox_to_anchor=(0., 1.), ncol=3, loc="lower left", borderaxespad=0.,
        frameon=False, fontsize=14, title="Population size", title_fontsize=14)
    ax.get_legend()._legend_box.align="left"
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(ax.get_yticks(), size = 14)
    ax.set_xticklabels([landscape_labels[xtick.get_text()] for xtick in ax.get_xticklabels()], size = 16)
#    ax.set_yticks(fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(1)
    fig.tight_layout()
    plt.show()

def plot_figure_2A(nearly_neutral, pr=1):
    condition = (((nearly_neutral["kind"]=="both") |
                (nearly_neutral["kind"]=="no mistranslation") ) &
                ((nearly_neutral["pr_num"] == str(pr)) |
                (nearly_neutral["pr_num"] == "Nan") ) &
                (nearly_neutral["N"] == r"10$^{4}$") )
    subset = nearly_neutral.loc[condition]
    subset = subset.reset_index()
    subset = subset.replace("both", "with mistranslation")
    subset = subset.replace("no mistranslation", "without mistranslation")
    subset.loc[:,"logSize"] = np.log10(subset.loc[:,"size"])
    order = ["without mistranslation", "with mistranslation"]
    landscape2 = []
    for row in range(subset.shape[0]):
        k = subset.loc[row, "landscape"]
        if k == "GB1":
            landscape2.append("Antibody-\nbinding")
        elif k=="ParD3E2":
            landscape2.append("Toxin-\nantitoxin\n(E2)")
        elif k=="ParD3E3":
            landscape2.append("Toxin-\nantitoxin\n(E3)")
        else:
            raise RuntimeError("Unknown landscape!")
    subset["landscape2"] = landscape2
    fig = plt.figure(figsize=(4.8,3.6))
    ax = sns.boxplot(x="landscape2", y="logSize", hue="kind", data=subset,
                hue_order=order, palette=[mplcolors.rgb2hex(wes(0.0)),
                mplcolors.rgb2hex(wes(0.9999))],
                saturation=1,
                showmeans=True,
                meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"})
    ax.set_ylabel(r"Network size (log$_{10}$)", fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.get_legend().set_title(None)
    plt.legend(fontsize=13, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_figure_2B(nearly_neutral, pr=1):
    medians = []
    means = []
    condition = ((nearly_neutral["landscape"]=="GB1") &
                ((nearly_neutral["pr_num"] == str(pr)) | (nearly_neutral["pr_num"] == "Nan"))  &
                (nearly_neutral["N"] == r"10$^{4}$")
                )
    subset = nearly_neutral.loc[condition]
    subset = subset.reset_index()
    subset = subset.replace("both", "mistranslation")
    subset.loc[:,"logSize"] = np.log10(subset.loc[:,"size"])
    for kind in subset["kind"].unique():
        mediansize = subset.loc[subset["kind"]==kind, "size"].median()
        meansize = subset.loc[subset["kind"]==kind, "size"].mean()
        medians.append([kind, mediansize])
        means.append([kind, meansize])
    fig = plt.figure(figsize=(6.5,3.6))
    ax = sns.boxplot(x="kind", y="logSize", data=subset, order=["no mistranslation",
        "Ne", "smooth", "mistranslation"], color="grey",
        saturation=1,
        showmeans=True,
        meanprops={"marker":"o",
               "markerfacecolor":"white",
               "markeredgecolor":"black",
              "markersize":"10"})
    ax.set_ylabel(r"Network size (log$_{10}$)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticklabels(["no mistranslation", "drift only", "fitness only", "mistranslation"], fontsize=14)
    ax.set_xlabel("")
#    ax.get_legend().set_title(None)
    plt.tight_layout()
    plt.show()
    return medians, means

def num_synonymous_members(fname="../results/neutral_networks/gb1/mistranslation_1_1e4.pkl"):
    with open(fname, "rb") as f:
        networks = pickle.load(f)
        neutral = restructure_networks(networks)
    polyptnum = []
    for nw,members in neutral.items():
        aaseq = []
        for nt in members:
            aaseq.append(evo.translate(nt))
        polyptnum.append(len(set(aaseq)))
    return polyptnum

def neighbourhood_fixprob(results):
    # Do genotypes have different neighbourhoods/effect of mistranslation
    ord = results.sort_values(by=["fit_wt"])
    sns.boxplot(y="wt", x="fixprob", data=ord, orient="h"); plt.show()
    ord = results.sort_values(by=["fit_wt_mis"])
    sns.boxplot(y="wt", x="fixprob_mis", data=ord, orient="h"); plt.show()


################################################################################
#                  A D A P T I V E    W A L K S                                #
################################################################################

def read_lastline(fname):
    with open(fname, 'r') as f:
        lastline = f.readlines()[-1]
    return [float(n) for n in lastline.split(" ")]

def read_lastline_seq(fname):
    with open(fname, 'r') as f:
        lastline = f.readlines()[-1]
    return lastline.split(" ")[0]

def get_adaptive_walk_endpoints(directory):
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    end = {}
    for f in flist:
        seq = f.split("_")[0]
        end[seq] = read_lastline(directory+f)[0]
    # sort to make sure order is the same for all datasets
    seqs = sorted([s for s in end.keys()])
#    sequences = []
    ordend = []
    for seq in seqs:
        ordend.append(end[seq])
    return ordend

def get_adaptive_walk_sequence_endpoints(directory):
    flist = [f for f in os.listdir(directory) if "walk" in f]
    end = {}
    for f in flist:
        seq = f.split("_")[0]
        end[seq] = read_lastline_seq(directory+f)
    # sort to make sure order is the same for all datasets
    seqs = sorted([s for s in end.keys()])
#    sequences = []
    ordend = []
    for seq in seqs:
        ordend.append(end[seq])
    return ordend

def nonparametric_test_difference(nomisend, misend):
    meandiff = []
    comb = nomisend + misend
    for i in range(10000):
        sample = np.random.choice(comb, size=(len(nomisend), 2))
        meandiff.append(np.mean(sample[:,0]) - np.mean(sample[:,1]))
    ax = plt.hist(meandiff, density=True)
    plt.vlines(np.mean(nomisend)-np.mean(misend), 0, 1.1*max(ax[0]), color="red")
    plt.xlabel("Difference in mean fitness")
    plt.ylabel("Frequency")
    plt.ylim(0, 1.1*max(ax[0]))
    plt.show()
    return meandiff

def plot_adaptive_walk_endpoints(nomisend, misend, save_figure=None):
    # Mistranslation has no statiscally significant effect on the fitnesses reached
    # at the end of the simulations
    print(stats.wilcoxon(nomisend, misend))
    # Given the final fitnesses in the simulation, plot a histogram
    plt.hist([nomisend, misend], bins=8, color=[wes(0.0), wes(0.9999)],
        label=["Without mistranslation", "With mistranslation"])
    plt.xlabel("Absolute fitness")
    plt.ylabel("Frequency")
    plt.xlim(0, max(nomisend))
    plt.legend(frameon=False)
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def get_adaptive_walk_endpoints_all(landscape="gb1", directory="../results/adaptive_walk/"):
    fitnesses = []
    for n in [4,6,8]:
        noend = get_adaptive_walk_endpoints(directory+landscape+"/no_mistranslation_popsize1e{}/".format(n))
        for fit in noend:
            fitnesses.append([r"10$^{"+str(n)+"}$", n, "without mistranslation", fit])
        for pr in [1,500]:
            misend = get_adaptive_walk_endpoints(directory+landscape+"/proteins{}_popsize1e{}/".format(pr,n))
            for fit in misend:
                if pr == 1:
                    kind = "mistranslation, 1 protein"
                else:
                    kind = "mistranslation, 500 proteins"
                fitnesses.append([r"10$^{"+str(n)+"}$", n, kind, fit])
    fitnesses = pd.DataFrame(fitnesses, columns=["Population size", "n", "kind", "Fitness at end of walk"])
    return fitnesses

def get_maximum_fitnesses(landscape="gb1"):
    landnomis = evo.read_fitness_dictionary("../data/"+landscape+".txt")
    maxnomis = max([v for v in landnomis.values()])
    db1 = sql.connect("../data/"+landscape+"_1.db")
    cur1 = db1.cursor()
    cur1.execute("SELECT MAX(fmean) FROM {}".format(landscape))
    maxmis1 = cur1.fetchone()[0]
    db500 = sql.connect("../data/"+landscape+"_500.db")
    cur500 = db1.cursor()
    cur500.execute("SELECT MAX(fmean) FROM {}".format(landscape))
    maxmis500 = cur500.fetchone()[0]
    return [maxnomis, maxmis1, maxmis500]

def test_adaptive_walk_endpoints(fitnesses, n=4, landscape="gb1"):
    maxnomis, maxmis1, maxmis500 = get_maximum_fitnesses(landscape)
    nomis = fitnesses.loc[(fitnesses["n"]==n) & (fitnesses["kind"]=="without mistranslation"), "Fitness at end of walk"].to_numpy()
    mis1 = fitnesses.loc[(fitnesses["n"]==n) & (fitnesses["kind"]=="mistranslation, 1 protein"), "Fitness at end of walk"].to_numpy()
    mis500 = fitnesses.loc[(fitnesses["n"]==n) & (fitnesses["kind"]=="mistranslation, 500 proteins"), "Fitness at end of walk"].to_numpy()
    percno = nomis / maxnomis
    perc1 = mis1 / maxmis1
    perc500 = mis500 / maxmis500
    print(stats.kruskal(percno, perc1, perc500))
    print(dunn([percno, perc1, perc500], p_adjust="bonferroni"))
    return percno, perc1, perc500

def plot_figure_4AB(fitnesses):
    order = ["without mistranslation", "mistranslation, 500 proteins", "mistranslation, 1 protein"]
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = sns.boxplot(x="Population size", y="Fitness at end of walk", hue="kind", data=fitnesses,
            hue_order=order,
            palette=[mplcolors.rgb2hex(wes(0.9999)), mplcolors.rgb2hex(bluescale(0.1)),
            mplcolors.rgb2hex(redscale(1.0))], saturation=1,
            showmeans=True,
            meanprops={"marker":"o",
                   "markerfacecolor":"white",
                   "markeredgecolor":"black",
                  "markersize":"10"})
    ax.get_legend().set_title(None)
    plt.xlabel(ax.get_xlabel(), fontsize=16)
    plt.ylabel(ax.get_ylabel(), fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0., 1.), loc="lower left", borderaxespad=0.,
        frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()

def test_differences_walk_endpoints(fitnesses, n=4):
    N = r"10$^{"+str(n)+"}$"
    nomis = fitnesses.loc[(fitnesses["Population size"]==N) & (fitnesses["kind"]=="without mistranslation"), "Fitness at end of walk"]
    mis1 = fitnesses.loc[(fitnesses["Population size"]==N) & (fitnesses["kind"]=="mistranslation, 1 protein"), "Fitness at end of walk"]
    mis500 = fitnesses.loc[(fitnesses["Population size"]==N) & (fitnesses["kind"]=="mistranslation, 500 proteins"), "Fitness at end of walk"]
    print("Standard deviations")
    print("no mistranslation: ", np.std(nomis))
    print("mistranslation, 1: ",  np.std(mis1))
    print("mistranslation, 500: ", np.std(mis500))
    print("Medians")
    print("no mistranslation: ", np.median(nomis))
    print("mistranslation, 1: ",  np.median(mis1))
    print("mistranslation, 500: ", np.median(mis500))
    print("Means")
    print("no mistranslation: ", np.mean(nomis))
    print("mistranslation, 1: ",  np.mean(mis1))
    print("mistranslation, 500: ", np.mean(mis500))
    print(stats.wilcoxon(nomis, mis1))
    print(stats.wilcoxon(nomis, mis500))
    print(stats.friedmanchisquare(nomis, mis1, mis500))
    data = np.array([nomis, mis1, mis500]).T
    print(nemenyi(data))

def endpoint_fitness_peak(endpoints, landscape, Ne):
    # Check if the endpoints are in fact on a fitness peak
    network_kind = {} # Is the sequence on a peak, a trough, or a saddle?
    network_size = {}
    network_seq = {}
    network_neigh = {}
    if type(landscape) == dict: # Order the landscape from fittest to least fit genotype
        ordlandscape = OrderedDict()
        for k,v in sorted(landscape.items(), key = lambda item: item[1], reverse=True):
            ordlandscape[k] = v
        landscape = ordlandscape
    for seq in np.unique(endpoints):
        neutral = {}
        non_neutral = []
        mutkind = []
        neutral[seq] = False # This sequence has not yet been assessed
        while not all([v == True for v in neutral.values()]):
            new_seqs = []
            new_edges = [] # New non-neutral edges
            new_mtkind = []
            for k in neutral.keys():
                if not neutral[k]:
                    neutral[k] = True
                    nn, not_nn, mtkind = find_neutral_neighbours(landscape, k, Ne)
                    new_seqs.extend(nn)
                    new_edges.extend(not_nn)
                    new_mtkind.extend(mtkind)
            for newseq in new_edges:
                non_neutral.append(newseq)
            for newseq in new_seqs:
                if not newseq in neutral.keys():
                    neutral[newseq] = False
            for newseq in new_mtkind:
                mutkind.append(newseq)
        network_neigh[seq] = non_neutral
        network_seq[seq] = [k for k in neutral.keys()]
        network_size[seq] = len(neutral)
        if all([v == "beneficial" for v in mutkind]):
            network_kind[seq] = "trough"
        elif all([v == "deleterious" for v in mutkind]):
            network_kind[seq] = "peak"
        else:
            network_kind[seq] = "saddle"
    return network_kind, network_size, network_seq, network_neigh

def get_prop_mistranslated(seq, misrates=evo.misrates):
    probs = []
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        rates = misrates[codon]
        # Probability of not mistranslating for one amino acid position
        probs.append(1 - np.sum([r[0] for r in rates.values()]))
    # Probability of mistranslation
    mispr = 1 - np.product(probs)
    return mispr

def get_adaptive_walk_endpoints_percent(nomisend, misend, landscape, mislandscape):
    #What percentage of sequences have a higher fitness than the endpoint?
    landscape.execute("SELECT COUNT(*) FROM gb1nomis")
    totseq = landscape.fetchone()[0]
    nomisprc = []
    misprc = []
    for fit in tqdm(nomisend):
        landscape.execute("SELECT COUNT(*) FROM gb1nomis WHERE fitness > {}".format(fit))
        nomisprc.append(landscape.fetchone()[0]/totseq)
    for fit in tqdm(misend):
        mislandscape.execute("SELECT COUNT(*) FROM gb1 WHERE fmean > {}".format(fit))
        misprc.append(mislandscape.fetchone()[0]/totseq)
    return nomisprc, misprc

def plot_adaptive_walk_endpoints_percent(nomisprc, misprc):
    plt.hist([nomisprc, misprc], bins=50, density="True", label=["low", "high"])
    plt.xlabel("Percentage of sequences above final fitness")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def get_fixation_kind(directory, nsteps=int(1e5)):
    flist = [f for f in os.listdir(directory) if "walk" in f]
    num_synon = np.zeros((nsteps,))
    num_nonsyn = np.zeros((nsteps,))
    for f in flist:
        data = np.loadtxt(directory+f, dtype=str)
        mutsum = 0
        for i in range(len(data)-1):
            mutsum += int(data[i,1])
            prev = data[i,0]
            mut = data[i+1,0]
            if evo.hamming(evo.translate(prev), evo.translate(mut))==0:
                num_synon[mutsum] += 1
            else:
                num_nonsyn[mutsum] += 1
    return [num_nonsyn, num_synon]

def plot_fixation_kind(nomis_count, mis_count):
    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    for i,(data, mis) in enumerate(zip([nomis_count, mis_count], ["without", "with"])):
        for num, label, a in zip(data, ["nonsynonymous", "synonymous"], [1,0.2]):
            xpos = []
            ypos = []
            for j,c in enumerate(num):
#                if c > 0:
                xpos.append(j)
                ypos.append(c)
            fit = lowess(ypos, xpos, frac=1/5)
            #ax[i].scatter(xpos, ypos, alpha=a, label=label, edgecolors='none')
            ax[i].plot(fit[:,0], fit[:,1], label=label)
            ax[i].set_title(mis+" mistranslation", loc="left")
    plt.xlabel("Number of mutations")
    plt.ylabel("Number of fixations")
#    plt.xscale("log")
#    plt.yscale("log")
    ax[0].legend()
    plt.show()

def get_fnoise_endpoints(directory, mislandscape):
    # How robust are the endpoints? One measure is the fitness variance of the
    # endpoint sequences *as if* they are expressed at high mistranslation
    # Because the variance is directly related to the variance in fitness of a
    # sequence' neighbourhood
    # Could also look at neighbouring sequence fitness w/out mistranslation directly
    flist = [f for f in os.listdir(directory) if "walk" in f]
    nomisfnoise = []
    misfnoise = []
    nomisfmean = []
    misfmean = []
    nomisprc = []
    misprc = []
    for f in sorted(flist):
        if "True" in f:
            seq = read_lastline_seq(directory+f)
            mislandscape.execute("SELECT fmean,fvar FROM gb1 WHERE seq = ? ", (seq,))
            fmean,fvar = mislandscape.fetchone()
            misfnoise.append(fvar/fmean**2)
            misfmean.append(fmean)
            misprc.append(get_prop_mistranslated(seq)*100)
        else:
            seq = read_lastline_seq(directory+f)
            mislandscape.execute("SELECT fmean,fvar FROM gb1 WHERE seq = ? ", (seq,))
            fmean,fvar = mislandscape.fetchone()
            nomisfnoise.append(fvar/fmean**2)
            nomisfmean.append(fmean)
            nomisprc.append(get_prop_mistranslated(seq)*100)
    return nomisfnoise, misfnoise, nomisfmean, misfmean, nomisprc, misprc

def boxplot_mistranslation_prob_endpoints(nomisprc, misprc):
#    box = plt.boxplot([nomisprc, misprc], patch_artist=True,
#        labels=["Without mistranslation", "With mistranslation"])
#    for patch, color in zip(box['boxes'], [wes(0.9999), wes(0.0)]):
#        patch.set_facecolor(color)
    misprob = [[v, "Without mistranslation"] for v in nomisprc] + [[v, "With mistranslation"] for v in misprc]
    misprob = pd.DataFrame(misprob, columns=["Percentage mistranslated", "mis"])
    sns.boxplot(x="mis", y="Percentage mistranslated", data=misprob,
        palette=[mplcolors.rgb2hex(wes(0.0)), mplcolors.rgb2hex(wes(0.9999))])
    plt.xlabel("")
    plt.legend(frameon=False)
    plt.show()

def expand(cond):
    traj = np.empty(np.sum([int(v) for v in cond[:,1]]), dtype=cond.dtype)
    pos = 0
    for i in range(cond.shape[0]):
        rep = int(cond[i,1])
        newvals = np.repeat(cond[i,0], rep)
        traj[pos:pos+rep] = newvals
        pos += rep
    return traj

def get_mean_fitness_trajectory(directory, steps=1e5):
    # Too memory intensive to store all data points
    # Calculate mean first, then standard error
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    meantr = np.zeros(int(steps))
    for f in sorted(flist):
        read = np.loadtxt(directory+f)
        traj = expand(read)
        meantr = np.add(meantr, traj)
    meantr = meantr / len(flist)
#    sumsq = np.zeros(int(steps))
#    for f in sorted(flist):
#        read = np.loadtxt(directory+f)
#        traj = expand(read)
#        sumsq = np.add(sumsq, (traj - meantr)**2)
#    vartr = sumsq / (len(flist)-1)
#    stderr = np.sqrt(vartr) / np.sqrt(meantr)
    return meantr#, stderr]

def get_fitnesses_at_timepoint(directory, timepoint=50):
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    fits = []
    for f in sorted(flist):
        read = np.loadtxt(directory+f)
        traj = expand(read)
        fits.append(traj[timepoint])
    return np.array(fits)

def test_fitnesses_at_timepoint(landscape="gb1", timepoint=50):
    maxno, max1, max500 = get_maximum_fitnesses(landscape)
    fno = get_fitnesses_at_timepoint("../results/adaptive_walk/"+landscape+"/no_mistranslation_popsize1e4/",timepoint)
    f1 = get_fitnesses_at_timepoint("../results/adaptive_walk/"+landscape+"/proteins1_popsize1e4/",timepoint)
    f500 = get_fitnesses_at_timepoint("../results/adaptive_walk/"+landscape+"/proteins500_popsize1e4/",timepoint)
    fno = fno / maxno
    f1 = f1 / max1
    f500 = f500 / max500
    print(np.mean(fno), np.std(fno))
    print(np.mean(f1), np.std(f1))
    print(np.mean(f500), np.std(f500))
    print(stats.kruskal(fno, f1, f500))
    print(dunn([fno, f1, f500],p_adjust="bonferroni"))

def plot_figure_4CD(notr, mis1tr, mis500tr, save_figure=None, maxfit=8.8):
    labels = ["without mistranslation", "mistranslation, 1 protein", "mistranslation, 500 proteins"]
    fig = plt.figure(figsize=(4.8,4.8))
    for traj, mis in zip([notr, mis1tr, mis500tr], labels):
        mean = traj
        if mis.startswith("mistr") and "1" in mis:
            c = wes(0.9999)
        elif mis.startswith("mistr") and "500" in mis:
            c = bluescale(0.1)
        else:
            c = wes(0.0)
#        plt.fill_between(range(len(mean)), mean-1.96*stderr, mean+1.96*stderr, alpha=0.25,
#        color=c)
        plt.plot(range(len(mean)), mean, color=c, label=mis)
    plt.xlabel("Number of mutation-\nfixation events", fontsize=16)
    plt.ylabel("Fitness (arbitrary units)", fontsize=16)
    plt.xscale("symlog")
    plt.ylim(bottom=0, top=maxfit)
    plt.xlim(left=0, right=len(notr))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0., 1.), loc="lower left",
        frameon=False, borderaxespad=0., fontsize=14)
    plt.tight_layout()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def get_misrates_over_time(directory, nsteps=100, cutoff = np.inf):
    if cutoff < nsteps:
        raise InputError("Cutoff must be larger or equal to nsteps")
    flist = [f for f in os.listdir(directory) if "walk" in f]
    props = []
    knownseq = {}
    for f in sorted(flist):
        data = np.loadtxt(directory+f, dtype=str)
        seqs = data[:,0]
        times = np.zeros(len(seqs))
        for i in range(len(seqs)-1):
            times[i+1] = times[i] + float(data[i,1])
        # Cut off to focus only on the first steps
        seqs = seqs[times < cutoff]
        times = times[times < cutoff]
        stepsize = np.min((np.sum([int(v) for v in data[:,1]]), cutoff)) / nsteps
        for i in range(nsteps+1):
            timepoint = i * stepsize
            index = times <= timepoint
            s = seqs[index][-1]
            if s in knownseq.keys():
                mprop = knownseq[s]
            else:
                mprop = get_prop_mistranslated(s)
                knownseq[s] = mprop
            props.append([mprop, timepoint])
    return props

def get_mistranslation_rate_landscape(landscape="gb1_fitdict.txt"):
    fitdict = evo.read_fitness_dictionary("../data/"+landscape)
    misrates = []
    for aa in tqdm(fitdict.keys()):
        for nt in evo.reverse_translate(aa):
            misrates.append(get_prop_mistranslated(nt))
    return misrates

def get_percent_misrates_over_time(nomisprops, misprops, meanmisr, stdmisr):
    proptime = {}
    for vals,mis in zip([nomisprops, misprops], ["without mistranslation", "with mistranslation"]):
#        yraw = np.array([(v[0]-meanmisr)/stdmisr for v in vals])
        yraw = np.array([v[0]/meanmisr*100 for v in vals])
        xraw = np.array([v[1] for v in vals])
        x_pos = np.unique(xraw)
#        means = np.array([np.mean(yraw[xraw == v]) for v in np.unique(xraw)])
#        stderr = np.array([np.std(yraw[xraw == v]) for v in np.unique(xraw)])
        # bootstrapping using 50% of the data at one time
#        xgrid = np.linspace(0,int(np.sum([float(v) for v in data[:,1]])))
#        bootstrap = []
#        for i in range(100):
#            samples = np.random.choice(range(len(yraw)), len(yraw), replace=True)
#            ys = yraw[samples]
#            xs = xraw[samples]
#            smp_mean = np.array([np.mean(ys[xs == v]) for v in x_pos])
#            ysmooth = lowess(ys, xs, frac=frac, return_sorted=False)
#            ygrid = interpolate.interp1d(xs, ysmooth, fill_value='extrapolate')(xgrid)
#            bootstrap.append(smp_mean)
#        bootstrap = np.stack(bootstrap).T
#        means = np.mean(bootstrap, axis=0)
#        stderr = np.std(bootstrap, axis=0, ddof=0) / np.sqrt(means)
        means = np.array([np.mean(yraw[xraw==v]) for v in x_pos])
        stdv = np.array([np.std(yraw[xraw==v]) for v in x_pos])
        proptime[mis] = [x_pos, means, stdv/np.sqrt(means)]
    return proptime

def plot_figure_5C(proptime, save_figure=None, legend=True, ylim=(98.5,100.5)):
    fig = plt.figure(figsize=(4.8,3.6))
    for mis, vals in proptime.items():
        x_pos, means, stderr = vals
        if mis.startswith("without"):
            c = wes(0.9999)
        else:
            c = wes(0.0)
        plt.fill_between(x_pos, means-1.96*stderr, means+1.96*stderr, alpha=0.25,
            color=c)
#        plt.plot(xgrid, mean, color=c)
        plt.plot(x_pos, means, color=c, label=mis)
    plt.xlabel("Number of mutations", fontsize=16)
#    plt.ylabel("Deviation from mean\n"+r"mistranslation rate ($\sigma$)", fontsize=16)
    plt.ylabel("Percentage of mean\n"+r"mistranslation rate", fontsize=16)
    plt.xticks(np.linspace(0, max(x_pos), 5), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,max(x_pos))
    plt.ylim(*ylim)
    if legend:
        plt.legend(frameon=False, fontsize=14, bbox_to_anchor=(0., 1.), loc="lower left",)
#    plt.xscale("log")
    plt.tight_layout()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def get_effective_popsize_time(directory, landscape, steps=1e5):
    flist = [f for f in os.listdir(directory) if "walk" in f]
    N = float(directory[-4:-1])
    Netr = np.zeros(int(steps))
    landscape.execute("SELECT name FROM sqlite_master WHERE type='table'")
    name = landscape.fetchall()[0][0]
    for f in sorted(flist):
        read = np.loadtxt(directory+f, dtype=str)
        tr = []
        for row in read:
            landscape.execute("SELECT fmean,fvar FROM {} WHERE seq = ?".format(name), (row[0],))
            out = landscape.fetchall()
            fmean, fvar = out[0]
            Ne = evo.popsize_reduction(N, fvar / fmean**2)
            tr.append([Ne, int(row[1])])
        traj = expand(np.array(tr))
        Netr = np.add(Netr, traj)
    meanNetr = Netr / len(flist)
    return meanNetr

def get_change_fixation_probs(misdir, nomisdir, landscape, mislandscape):
    ch_nomis = []
    ch_mis = []
    flist = [f for f in os.listdir(nomisdir) if "walk" in f]
    misflist = [f for f in os.listdir(misdir) if "walk" in f]
    N = float(nomisdir[-4:-1])
    mislandscape.execute("SELECT name FROM sqlite_master WHERE type='table'")
    name = mislandscape.fetchall()[0][0]
    for f in sorted(flist):
        read = np.loadtxt(nomisdir+f, dtype=str)
        mislandscape.execute("SELECT fmean,fvar FROM {} WHERE seq IN ({})".format(name,",".join("?"*2)), read[:2,0])
        out = mislandscape.fetchall()
        ne, misup, s = evo.mistranslation_fixation_prob_from_fit(N, out)
        fwt, mwt, s, up=evo.no_mistranslation_fixation_prob(N, read[0,0], read[1,0], landscape)
        # How does the probability of the no mistranslation path change?
        ch_nomis.append(misup-up)
        # How does the probability of the mistranslation path change?
        names = f.split("_")
        names[1] = "True"
        read = np.loadtxt(misdir+"_".join(names), dtype=str)
        mislandscape.execute("SELECT fmean,fvar FROM {} WHERE seq IN ({})".format(name,",".join("?"*2)), read[:2,0])
        out = mislandscape.fetchall()
        ne, misup, s = evo.mistranslation_fixation_prob_from_fit(N, out)
        fwt, mwt, s, up=evo.no_mistranslation_fixation_prob(N, read[0,0], read[1,0], landscape)
        ch_mis.append(misup-up)
    return ch_nomis, ch_mis

def get_adaptive_walk_pseudotime(directory):
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    fit = []
    for f in sorted(flist):
        traj = np.loadtxt(directory+f)
        fit.append(traj[:,0])
    return fit

def plot_adaptive_walk_pseudotime(trajno, trajmis1, trajmis500, frac=1/4, save_figure=None):
    # First, get data in line between 0 (start) and 1 (end)
    for traj,mis in zip([trajno, trajmis1, trajmis500],
            ["without mistranslation", "with mistranslation, 1 protein", "with mistranslation, 500 proteins"]):
        xpos = []
        ypos = []
        for fit in traj:
            for i, f in enumerate(fit):
                xpos.append(i/(len(fit)-1))
                ypos.append(f)
        xraw = np.array(xpos)
        yraw = np.array(ypos)
        smoothed = lowess(yraw, xraw, frac=frac)
        # bootstrapping using 50% of the data at one time
        xgrid = np.linspace(0,1)
        bootstrap = []
        for i in range(100):
            samples = np.random.choice(range(len(yraw)), int(len(yraw)/2), replace=True)
            ys = yraw[samples]
            xs = xraw[samples]
            ysmooth = lowess(ys, xs, frac=frac, return_sorted=False)
#            ygrid = interpolate.interp1d(xs, ysmooth, fill_value='extrapolate')(xgrid)
            ygrid = interpolate.interp1d(xs, ysmooth)(xgrid)
            bootstrap.append(ygrid)
        bootstrap = np.stack(bootstrap).T
        mean = np.nanmean(bootstrap, axis=1)
        stderr = np.nanstd(bootstrap, axis=1, ddof=0)
        if mis.startswith("with ") and "1" in mis:
            c = wes(0.9999)
        elif mis.startswith("with ") and "500" in mis:
            c = bluescale(0.1)
        else:
            c = wes(0.0)
        plt.fill_between(xgrid, mean-1.96*stderr, mean+1.96*stderr, alpha=0.25,
            color=c)
        plt.plot(xgrid, mean, color=c, label=mis)
    plt.xlabel("Pseudotime")
    plt.ylabel("Fitness")
    plt.legend()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def get_first_fixation_fitness_change(directory):
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    fch = []
    time = []
    for f in sorted(flist):
        read = np.loadtxt(directory+f)
        fch.append(read[1,0]-read[0,0])
        time.append(read[0,1])
    return fch, time

def plot_first_fixation_fitness_change(fchno, fch1, fch500, save_figure=None, log=True):
    fitch = []
    labels = ["without", "1 protein", "500 proteins"]
    for fch, label in zip([fchno, fch1, fch500], labels):
        for ch in fch:
            fitch.append([label, ch])
    fitch = pd.DataFrame(fitch, columns = ["kind", "Change in fitness (arbitrary units)"])
    ax = sns.boxplot(x="kind", y="Change in fitness (arbitrary units)", data=fitch,
        palette=[mplcolors.rgb2hex(wes(0.)),
        mplcolors.rgb2hex(bluescale(1.0)), mplcolors.rgb2hex(bluescale(0.1))],
        showmeans=True,
        meanprops={"marker":"o",
               "markerfacecolor":"white",
               "markeredgecolor":"black",
              "markersize":"10"})
    ax.set_xlabel("")
    ax.yaxis.label.set_size(12)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=12)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(6e-7, 9)
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def get_stepnum_to_first_fixation(directory):
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    first = []
    for f in sorted(flist):
        read = np.loadtxt(directory+f)
        first.append(read[0,1])
    return first

def plot_stepnum_to_first_fixation(stepno, stepmis1, stepmis500, save_figure=None):
    fitch = []
    labels = ["without", "1 protein", "500 proteins"]
    for step, label in zip([stepno, stepmis1, stepmis500], labels):
        for s in step:
            fitch.append([label, s])
    fitch = pd.DataFrame(fitch, columns = ["kind", "Number of mutation-fixation events"])
    ax = sns.boxplot(x="kind", y="Number of mutation-fixation events", data=fitch,
        palette=[mplcolors.rgb2hex(wes(0.)),
        mplcolors.rgb2hex(bluescale(1.0)), mplcolors.rgb2hex(bluescale(0.1))])
    ax.set_xlabel("")
    ax.set_yscale("log")
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def get_stepnum_to_endpoint(directory, nsteps=1e5):
    # Simulations with mistranslation take many more steps to reach their endpoint
    # than simulations without mistranslation
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    last = []
    for f in sorted(flist):
        # number of simulation steps until endpoint: total - last
        last.append(nsteps - read_lastline(directory+f)[1])
    return last

def plot_histogram_stepnum_to_endpoint(lastfalse, lasttrue, nsteps=1e5):
    print(stats.wilcoxon(lastfalse, lasttrue))
    plt.hist([lastfalse, lasttrue], bins=50, density=False,
        label=["without mistranslation", "with mistranslation"],
        color=[wes(0.0), wes(0.9999)])
    plt.xlabel("Number of simulation steps to last fixation")
    plt.ylabel("Frequency")
    plt.legend()
    plt.yscale("log")
    plt.show()

def get_mutnum_to_endpoint(directory):
    flist = [f for f in os.listdir(directory) if "walk" in f]
    num = []
    for f in sorted(flist):
        # number of fixation events until endpoint: length of file
        num.append(len(np.loadtxt(directory+f, dtype=str))-1)
    return num

def plot_figure_5A(numfalse, numtrue):
    print(stats.wilcoxon(numfalse, numtrue))
    fig = plt.figure(figsize=(4.8,3.6))
    plt.hist([numfalse, numtrue], density=False, label=["without mistranslation", "with mistranslation"],
                color=[wes(0.9999), wes(0)])
    plt.xlabel("Number of fixation events", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False, bbox_to_anchor=(0., 1.), loc="lower left",)
    plt.tight_layout()
    plt.show()

def get_aachange_to_endpoint(directory):

    def _num_aa_change(ntseqs):
        # Take the nucleotide seq & calculate the number of aa changes
        pept = []
        aachange = 0
        aa = evo.translate(ntseqs[0])
        for i in range(1, len(ntseqs)):
            aa_new = evo.translate(ntseqs[i])
            if aa == aa_new:
                pass
            else:
                aachange += 1
            aa = aa_new
        return aachange

    flist = [f for f in os.listdir(directory) if "walk" in f]
    aach= []
    for f in sorted(flist):
        walk = np.loadtxt(directory+f, dtype=str)
        if len(walk.shape) > 1:
            aachange = _num_aa_change(walk[:,0])
        else:
            aachange = 0
        aach.append(aachange)
    return aach

def histogram_aachange_to_endpoint(aafalse, aatrue):
    print(stats.wilcoxon(aafalse, aatrue))
    plt.hist([aafalse, aatrue], bins = 50, density=True, label=["low", "high"])
    plt.xlabel("Number of polypeptide sequence changes")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def get_fitness_change_fixation(directory):
    # Upon fixation, the fitness increase tends to be larger without mistranslation
    # than with mistranslation
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    chfit = []
    for f in sorted(flist):
        traj = np.loadtxt(directory+f)
        # Average change in fitness per fixation event
        if len(traj.shape) > 1:
            chf = np.mean([traj[i+1][0]-traj[i][0] for i in range(len(traj)-1)])
        else:
            chf = 0
        chfit.append(chf)
    return chfit

def histogram_fitness_change_fixation(chfitfalse, chfittrue):
    print(stats.wilcoxon(chfitfalse, chfittrue))
    plt.hist([chfitfalse, chfittrue], density=True, bins=50, label=["low", "high"])
    plt.xlabel("Mean change in fitness")
    plt.ylabel("Density")
    plt.legend()
    plt.yscale("log")
    plt.show()

def get_fitness_changes_time(directory):
    # For those runs where more than one mutation became fixed
    # check if the first fixation event is larger for mistr or no mistr
    # Also compare the fitness benefits of the last fixation event
    # Or try to plot the whole trajectory in between??
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    chfit = []
    mutnum = []
    synon = []
    misrates = []
    sequences = []
    fitbeg = []
    fitend = []
    for f in sorted(flist):
        traj = np.loadtxt(directory+f)
        fitbeg.append(traj[0,0])
        fitend.append(traj[-1,0])
        seq = np.loadtxt(directory+"_".join(f.split("_")[:-1])+"_walk.txt", dtype=str)
        # bring sequences and fitnesses and timepoint in sync (fitnesses do not record
        # purely neutral changes)
        if len(seq) > 1:
            time = 0
            data = [] # time, sequence, fitness
            i = 0
            j = 0
            remainder = 0
            while i < len(seq):
                if int(seq[i,1])+remainder == int(traj[j,1]):
                    data.append([time, seq[i,0], traj[j,0]])
                    time += int(seq[i,1])
                    i += 1
                    j += 1
                    remainder = 0
                elif int(seq[i,1])+remainder < int(traj[j,1]): # due to synonymous mutation with same fitness
                # in this case, traj should always be bigger than seq
                    data.append([time, seq[i,0], traj[j,0]])
                    time += int(seq[i,1])
                    remainder += int(seq[i,1])
                    i += 1
                else:
                    raise RuntimeError("More entries in traj than seq?!")
            sy = []
            chf = []
            times = []
            misr = []
            seqs = []
            for row in range(len(data)-1):
                times.append(data[row+1][0])
                chf.append(data[row+1][2] - data[row][2])
                misr.append([get_prop_mistranslated(data[row][1]), get_prop_mistranslated(data[row+1][1])])
                prev = data[row][1]
                mut = data[row+1][1]
                if evo.hamming(evo.translate(prev), evo.translate(mut))==0:
                    sy.append(True)
                else:
                    sy.append(False)
                seqs.append(data[row][1])
            seqs.append(data[-1][1])
            # Get all runs toegether
            synon.append(sy)
            chfit.append(chf)
            mutnum.append(times)
            misrates.append(misr)
            sequences.append(seqs)
    return chfit, mutnum, synon, misrates, sequences, fitbeg, fitend

def evaluate_percentage_change(chfit, synon, fitbeg, fitend):
    nonsyn = []
    syn = []
    for chf, sy, fb, fe in zip(chfit, synon, fitbeg, fitend):
        nonsynch = 0
        synch = 0
        for ch, s in zip(chf, sy):
            if s:
                synch += ch/(fe-fb)
            else:
                nonsynch += ch/(fe-fb)
        nonsyn.append(nonsynch)
        syn.append(synch)

    print("Num nonsynonmymous: {}".format(len(nonsyn)))
    print("Num synonymous: {}".format(len(syn)))
    print("mean nonsynonymous")
    print(np.mean(nonsyn)*100)
    print("std nonynonymous")
    print(np.std(nonsyn)*100)
    print("mean synonymous")
    print(np.mean(syn)*100)
    print("std synonymous")
    print(np.std(syn)*100)
    print(stats.wilcoxon(nonsyn, syn))

def get_synon_nonsynon(directory):
    flist = [f for f in os.listdir(directory) if "walk" in f]
    synon = []
    for f in sorted(flist):
        seq = np.loadtxt(directory+f, dtype=str)
        if len(seq) > 1:
            sy = []
            for i in range(len(seq)-1):
                prev = seq[i,0]
                mut = seq[i+1,0]
                if evo.hamming(evo.translate(prev), evo.translate(mut))==0:
                    sy.append(True)
                else:
                    sy.append(False)
                synon.append(sy)
    return synon

def get_correlation_synon_nonsynon(directory):
    synon = get_synon_nonsynon(directory)
    synnum = []
    nonnum = []
    for row in synon:
        s = sum([v==True for v in row])
        n = sum([v==False for v in row])
        synnum.append(s)
        nonnum.append(n)
    print(stats.kendalltau(synnum, nonnum))
    plt.scatter(synnum, nonnum)
    plt.show()

def get_neutrality_time(sequences, N, landscape):
    neutral = []
    if type(landscape) != dict:
        landscape.execute("SELECT name FROM sqlite_master WHERE type='table'")
        name = landscape.fetchall()[0][0]
    else:
        Ne = N
    for walk in sequences:
        neut = []
        for i in range(len(walk)-1):

            if type(landscape) != dict:
                landscape.execute("SELECT fmean,fvar FROM {} WHERE seq IN ({})".format(name,",".join("?"*2)), (walk[i], walk[i+1],))
                out = landscape.fetchall()
                Ne, up, s = evo.mistranslation_fixation_prob_from_fit(N, out)
            else:
                f_wt = landscape[evo.translate(walk[i])]
                f_mt = landscape[evo.translate(walk[i+1])]
                s = f_mt/f_wt - 1
            if abs(Ne*s) < 0.25:
                neut.append(True)
            else:
                neut.append(False)
        neutral.append(neut)
    return neutral

def plot_figure_5B(chfit, mutnum, synon):
    syn = []
    non = []
    color = []
    for tser,fser,sy in zip(mutnum, chfit, synon):
        for t,f,s in zip(tser, fser, sy):
            if s:
                syn.append([t,f])
            else:
                non.append([t,f])
    syn = np.array(syn)
    non = np.array(non)
    fig, ax = plt.subplots(2,1, sharex=True, sharey=False, figsize=(6.24,7.2))
    ax[0].scatter(non[:,0], non[:,1], alpha=0.4, edgecolor='none', marker='.', color=greenbrown(0.9999))
    ax[1].scatter(syn[:,0], syn[:,1], alpha=0.4, edgecolor='none', marker='.', color=greenbrown(0.))
#    plt.xscale("log")
#    plt.yscale("symlog")
    legend_elem = [Line2D([0], [0], marker='.', color='w', label='Nonsynonymous',
                          markerfacecolor=greenbrown(0.9999), markersize=12),
                    Line2D([0], [0], marker='.', color='w', label='Synonymous',
                         markerfacecolor=greenbrown(0.), markersize=12),
                          ]
    ax[0].legend(handles=legend_elem, frameon=False, fontsize=14)
    plt.xlabel("Number of mutations", fontsize=16)
    plt.xlim(-500,1e5+500)
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    fig.text(1e-2, 0.5, 'Fitness change upon fixation', va='center', rotation='vertical', size=16)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15)
    plt.show()

def evaluate_misrates_effect_synonymous(chfit, synon, misrates, cutoff=0.002):
    topsynfch = []
    topsynmch = []
    lowsynfch = []
    lowsynmch = []
    synfch = []
    synmch = []
    nsynfch = []
    nsynmch = []
    for fch, sy, misr in zip(chfit, synon, misrates):
        for f,s,m in zip(fch, sy, misr):
            if s and f > cutoff:
                topsynfch.append(f)
                topsynmch.append( (m[1]-m[0])/m[0] )
            elif s and f < cutoff:
                lowsynfch.append(f)
                lowsynmch.append( (m[1]-m[0])/m[0] )
            if s:
                synfch.append(f)
                synmch.append( (m[1]-m[0])/m[0] )
            else:
                nsynfch.append(f)
                nsynmch.append( (m[1]-m[0])/m[0] )
    print("Large effect synonymous mutations")
    print(stats.kendalltau(topsynfch, topsynmch))
    print("N={}".format(len(topsynfch)))
    print("Small effect synonymous mutations")
    print(stats.kendalltau(lowsynfch, lowsynmch))
    print("N={}".format(len(lowsynfch)))
    print("Synonymous mutations")
    print(stats.kendalltau(synfch, synmch))
    print("N={}".format(len(synfch)))
    print("Nonsynonymous mutations")
    print(stats.kendalltau(nsynfch, nsynmch))
    print("N={}".format(len(nsynfch)))
    print("Relative percentage change {}%".format(np.mean(topsynmch)*100))


def plot_change_in_fitness_trajectory(nomischfit, mis1chfit, mis500chfit, frac=1/5, save_figure=None):
    # First, get data in line between 0 (start) and 1 (end)
    labels = ["without mistranslation", "with mistranslation, 1 protein", "with mistranslation, 500 proteins"]
    for traj,mis in zip([nomischfit, mis1chfit, mis500chfit], labels):
        xpos = []
        ypos = []
        for chf in traj:
            if len(chf) > 1:
                for i, ch in enumerate(chf):
                    xpos.append(i/(len(chf)-1))
                    ypos.append(ch)
        xraw = np.array(xpos)
        yraw = np.array(ypos)
        smoothed = lowess(yraw, xraw, frac=frac)
        # bootstrapping using 50% of the data at one time
        xgrid = np.linspace(0,1)
        bootstrap = []
        for i in range(100):
            samples = np.random.choice(range(len(yraw)), int(len(yraw)/2), replace=True)
            ys = yraw[samples]
            xs = xraw[samples]
            ysmooth = lowess(ys, xs, frac=frac, return_sorted=False)
            ygrid = interpolate.interp1d(xs, ysmooth)(xgrid)
            bootstrap.append(ygrid)
        bootstrap = np.stack(bootstrap).T
        mean = np.nanmean(bootstrap, axis=1)
        stderr = np.nanstd(bootstrap, axis=1, ddof=0)
        if mis.startswith("with ") and "1" in mis:
            c = wes(0.9999)
        elif mis.startswith("with ") and "500" in mis:
            c= redscale(0.1)
        else:
            c = wes(0.0)
        plt.fill_between(xgrid, mean-1.96*stderr, mean+1.96*stderr, alpha=0.25,
            color=c)
        plt.plot(xgrid, mean, color=c, label=mis)
    plt.xlabel("Pseudotime")
    plt.ylabel("Change in fitness")
    plt.legend()
    if save_figure:
        plt.savefig(save_figure)
        plt.close()
    else:
        plt.show()

def get_neutral_network_mistranslation_probs(network_seqs, gb1, landscape,
        seq='ATTAGAGCTTGT', misrates=evo.misrates):
    data = []
    for s in network_seqs[seq]:
        landscape.execute("SELECT fmean FROM gb1 WHERE seq = ?", (s,))
        # Mean fitness with mistranslation
        fmis = landscape.fetchone()[0]
        # Fitness without mistranslation
        aa = evo.translate(s)
        fnomis = gb1[aa]
        # change in fitness due to mistranslation
        fitchange = fmis - fnomis
        # Probability of mistranslation
        mispr = get_prop_mistranslated(s, misrates)
        # Mean fitness of neighbours
        neigh = evo.find_neighbours(s)
        nfits = []
        for n in neigh:
            naa = evo.translate(n)
            if naa in gb1.keys():
                nfits.append(gb1[naa])
        data.append([s, fnomis, fmis, fitchange, mispr*100, np.mean(nfits)])
    return pd.DataFrame(data, columns=["seq", "fit", "fitmis", "fitch", "misperc","nfit"])

def plot_neutral_network_mistr_probs(fitchanges):
    sns.scatterplot(x="misperc", y="fitmis", color=wes(0.5), data=fitchanges)
    plt.xlabel("Percentage mistranslated")
    plt.ylabel("Fitness with mistranslation")
    plt.show()

def plot_figure_4E(directory="../results/peak_counting/", N="1e4"):
    peak = []
    landscape_labels = ["Antibody-\nbinding","Toxin-\nantitoxin\n(E2)","Toxin-\nantitoxin\n(E3)"]
    for l,ll in zip(["GB1", "ParD3E2", "ParD3E3"], landscape_labels):
        for m in ["no_mistrans_", "mistrans_1_", "mistrans_500_"]:
            fname = directory + "/{}/".format(l.lower()) + m +"{}_kind.pkl".format(N)
            with open(fname, "rb") as f:
                kind = pickle.load(f)
                count = sum([v=="peak" for v in kind.values()])
            if m == "no_mistrans_":
                pr = "without mistranslation"
            elif "1" in m:
                pr = "mistranslation, 1 protein"
            else:
                pr = "mistranslation, 500 proteins"
            peak.append([ll, pr, count])
    peak = pd.DataFrame(peak, columns=["landscape", "mistr", "Number of fitness peaks"])
    order = ["without mistranslation", "mistranslation, 500 proteins", "mistranslation, 1 protein"]
    fig = plt.figure(figsize=(4.8, 3.6))
    ax = sns.barplot(x="landscape", y="Number of fitness peaks", hue="mistr", hue_order=order,
            data=peak, palette=[mplcolors.rgb2hex(wes(0.)),
            mplcolors.rgb2hex(bluescale(0.1)), mplcolors.rgb2hex(bluescale(1.0))])
#    ax.get_legend().set_title(None)
    plt.legend([],[], frameon=False)
    ax.set_xlabel("")
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    #plt.legend(bbox_to_anchor=(0., 1.), loc="lower left", borderaxespad=0.,
    #    frameon=False, fontsize=14)
#    plt.setp(ax.get_legend().get_texts(), fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_figure_4F(directory="../results/epistasis/"):
    # Using code from https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
    landscape_labels = ["Antibody-\nbinding","Toxin-\nantitoxin\n(E2)","Toxin-\nantitoxin\n(E3)"]
    epiprop = []
    flist = [f for f in os.listdir(directory) if f.endswith("csv")]
    for f in sorted(flist):
        name = f.split("_")[1]
        if "no" in f:
            kind = "without mistranslation"
        elif "500" in f:
            kind = "mistranslation, 500 proteins"
        elif "1" in f:
            kind = "mistranslation, 1 protein"
        data = pd.read_csv(directory + f)
        reci = sum(data["kind"] == "reci") / data.shape[0] * 100
        sign = sum(data["kind"] == "sign") / data.shape[0] * 100
        magn = sum(data["kind"] == "magn") / data.shape[0] * 100
        epiprop.append( [name, kind, reci, sign, magn] )
    print(pd.DataFrame(epiprop, columns=["landscape", "kind", "reci", "sign", "magn"]))
    labels = []
    magnitude = []
    reciprocal = []
    signepist = []
    for row in epiprop:
        labels.append(", \n".join([row[0], row[1]]))
        reciprocal.append(row[2])
        signepist.append(row[3])
        magnitude.append(row[4])
    fig, ax = plt.subplots(figsize=(6.4,4.8))
    colors = [mplcolors.rgb2hex(wes(0.)), mplcolors.rgb2hex(bluescale(0.1)), mplcolors.rgb2hex(bluescale(1.0))]
    ax.bar(labels, magnitude, label="magnitude", color=colors*3)
    ax.bar(labels, signepist, label="sign", color=colors*3)
    ax.bar(labels, reciprocal, label="reciprocal",color=colors*3)

    h,l = ax.get_legend_handles_labels() # get the handles we want to modify
    for hatch,(i, pa) in zip([None, "/"*3, "x"*3], enumerate(h)):
        for j,rect in enumerate(pa.patches): # for each index
            if j % 3 == 0:
                off = 0.2
            elif j % 3 == 1:
                off = 0
            else:
                off = -0.2
            rect.set_x(rect.get_x() + off)
            rect.set_hatch(hatch) #edited part
                #rect.set_width(1 / float(n_df + 1))
    ax.set_xticks(np.arange(1, 9, 3))
    ax.set_xticklabels(landscape_labels, rotation = 0, fontsize=14)
    ax.tick_params(axis='y', which='major', labelsize=11)

    n=[]
    for i,H in enumerate([None, "/"*3, "x"*3]):
        n.append(ax.bar(0, 0, color="white", hatch=H, edgecolor="k"))
    pr=[]
    names = ["without mistranslation", "mistranslation, 500 proteins", "mistranslation, 1 protein"]
    for name,c in zip(names, colors):
        pr.append(ax.bar(0,0, color=c, label=name))

    l1 = ax.legend(pr, names, bbox_to_anchor=(0., 1.), loc="lower left", frameon=False,fontsize=14)
    l2 = plt.legend(n, ["magnitude", "sign", "reciprocal"], bbox_to_anchor=(0.65, 1.), loc="lower left", frameon=False,fontsize=14)
    ax.add_artist(l1)

    ax.set_ylabel("Fraction of samples (%)", fontsize=16)
#    ax.legend()
    plt.tight_layout()
    plt.show()

def mistranslation_influence_landscape(landscape="gb1_fitdict.txt", percentage=0.01):
    fitdict = evo.read_fitness_dictionary("../data/"+landscape)
    meanrates = {}
    for aa,codons in evo.codontable.items():
        if aa != "*":
            rates = []
            for c in codons:
                # sum the probabilities of mistranslation for each possible misincorporated aa
                rates.append(np.sum([v[0] for v in evo.misrates[c].values()]))
            # get the mean rate of mistranslation for each codon mapping to a given amino acid
            meanrates[aa] = np.mean(rates)
    # Correlate the fitness of a sequence with its mean rate of mistranslation
    fitn = []
    misr = []
    for aa in fitdict.keys():
        fitn.append(fitdict[aa])
        rates = []
        for a in aa:
            # rate of correct translation
            rates.append(1-meanrates[a])
        # mean rate of mistranslation for entire sequence
        misr.append(1-np.product(rates))
    print("Entire landscape")
    print(stats.kendalltau(fitn, misr))
    print("n={}".format(len(fitn)))
    top = sorted(fitn, reverse=True)[:int(len(fitn)*percentage)]
    topf = []
    topm = []
    for f,m in zip(fitn, misr):
        if f >= top[-1]:
            topf.append(f)
            topm.append(m)
    print("Top {}% of landscape".format(percentage*100))
    print(stats.kendalltau(topf, topm))
    print("n={}".format(len(topf)))
    return fitn, misr

def get_repeatability_endpoints(endpoints, fitdict):
    aaend = [evo.translate(s) for s in endpoints]
    nobs = [] # Number of walks ending at this amino acid sequence
    misraa = [] # Mean mistranslation rate of amnio acid sequence
    aafit = [] # Fitness of amino acid sequence (no mistranslation)
    ntscount = [] # Number of sequences that map to amino acid sequence
    for aa, n in Counter(aaend).items():
        nobs.append(n)
        aafit.append(fitdict[aa])
        nts = evo.reverse_translate(aa)
        ntscount.append(len(nts))
        p = [get_prop_mistranslated(s) for s in nts]
        misraa.append(np.mean(p))
    return nobs, misraa, aafit, ntscount

def get_waiting_times_between_fixations(directory, threshold=1e4):
    flist = [f for f in os.listdir(directory) if "walk" in f]
    # Load the appropriate landscape
    name = directory.split("/")[3]
    protnum = directory.split("/")[4].split("_")[0][8:]
    if "no_mistranslation" in directory:
        landscape = evo.read_fitness_dictionary("../data/"+name+".txt")
    else:
        db = sql.connect("../data/"+name+"_"+protnum+".db")
        landscape = db.cursor()
#    precseq = []
#    prevseq = []
#    nextseq = []
#    wait = [] # number of unsuccessful mutations before fixation
    syn = [] # synonymous or not?
    synprec = [] # was the preceding mutation synonymous or not?
    mutim = [] # number of mutations since beginning of simulation
    neighbeneprevsy = []
    neighbeneprevno = []
    neighbeneprecsy = []
    neighbeneprecno = []
#    neighbene = [] # change in the number of higher fitness neighbours
#    chmaxneigh = [] # change in the fitness of the fittest neighbour
    for f in tqdm(sorted(flist)):
        walk = np.loadtxt(directory+f, dtype=str)
        time = 0
        for i in range(1,len(walk)-2):
            # do not consider last mutation: the time it exists is only until
            # the end of the simulation and not the next mutation
            prec = walk[i-1,0] # for i = 0, this will retrieve the last entry of
            # the walk, but it will be ignored later as only data after mutation-
            #fixation events 1e4 are of interest
            prev = walk[i, 0]
#            prevseq.append(prev)
            mut = walk[i+1, 0]
#            nextseq.append(mut)
            w = int(walk[i, 1])
#            wait.append(w)
            time += int(walk[i, 1])
            mutim.append(time)
            if evo.hamming(evo.translate(prev), evo.translate(mut)) == 0:
                syn.append(True)
            else:
                syn.append(False)
            if evo.hamming(evo.translate(prec), evo.translate(prev)) == 0:
                synprec.append(True)
            else:
                synprec.append(False)
            numbeneprecsy = 0
            numbeneprecno = 0
            if not type(landscape) == dict:
                landscape.execute("SELECT fmean FROM {} WHERE seq = ?".format(name), (prec,))
                wt = landscape.fetchone()[0]
                seqs = evo.find_neighbours(prec)
                landscape.execute("SELECT seq,fmean FROM {} WHERE seq IN ({})".format(name, ",".join("?"*len(seqs))), seqs)
                fit = landscape.fetchall()
                for m in fit:
                    if m[1] > wt and evo.hamming(evo.translate(m[0]),evo.translate(prec))==0:
                        numbeneprecsy += 1
                    elif m[1] > wt:
                        numbeneprecno += 1
#                maxnprec = max([m[1] for f in fit]) / wt
            else:
                neighs = evo.find_neighbours(prec)
                for n in neighs:
                    if evo.translate(n) in landscape.keys():
                        s = evo.hamming(evo.translate(n),evo.translate(prec))==0
                        fitter = landscape[evo.translate(n)] > landscape[evo.translate(prec)]
                        if fitter and s:
                            numbeneprecsy += 1
                        elif fitter:
                            numbeneprecno += 1
#                maxnprec = max([landscape[evo.translate(n)] for n in neighs if evo.translate(n) in landscape.keys()]) / landscape[evo.translate(prec)]
            numbeneprevsy = 0
            numbeneprevno = 0
            if not type(landscape) == dict:
                landscape.execute("SELECT fmean FROM {} WHERE seq = ?".format(name), (prev,))
                wt = landscape.fetchone()[0]
                seqs = evo.find_neighbours(prev)
                landscape.execute("SELECT seq,fmean FROM {} WHERE seq IN ({})".format(name, ",".join("?"*len(seqs))), seqs)
                fit = landscape.fetchall()
                for m in fit:
                    if m[1] > wt and evo.hamming(evo.translate(m[0]),evo.translate(prev))==0:
                        numbeneprevsy += 1
                    elif m[1] > wt:
                        numbeneprevno += 1
#                maxnprev = max([m[1] for f in fit]) / wt
            else:
                neighs = evo.find_neighbours(prev)
                for n in neighs:
                    if evo.translate(n) in landscape.keys():
                        s = evo.hamming(evo.translate(n),evo.translate(prev))==0
                        fitter = landscape[evo.translate(n)] > landscape[evo.translate(prev)]
                        if fitter and s:
                            numbeneprevsy += 1
                        elif fitter:
                            numbeneprevno += 1
#                maxnprev = max([landscape[evo.translate(n)] for n in neighs if evo.translate(n) in landscape.keys()]) / landscape[evo.translate(prev)]
            neighbeneprevsy.append(numbeneprevsy)
            neighbeneprevno.append(numbeneprevno)
            neighbeneprecsy.append(numbeneprecsy)
            neighbeneprecno.append(numbeneprecno)
#            neighbene.append(numbeneprevsy+numbeneprevno-numbeneprecsy-numbeneprecno)
#            chmaxneigh.append(maxnprev-maxnprec)
#    waiting = [[pc,pr,mut,w,s,m,n,p,c,npsy,npno,ncsy,ncno] for pc,pr,mut,w,s,m,n,p,c,npsy,npno,ncsy,ncno in zip(precseq, prevseq, nextseq, wait, syn, mutim, neighbene, synprec,chmaxneigh,neighbeneprecsy, neighbeneprecno, neighbeneprevsy, neighbeneprevno) if m > threshold]
    waiting = [[s,m,p,npsy,npno,ncsy,ncno] for s,m,p,npsy,npno,ncsy,ncno in zip(syn, mutim,synprec,neighbeneprecsy, neighbeneprecno, neighbeneprevsy, neighbeneprevno) if m > threshold]
#    waiting = pd.DataFrame(waiting, columns=["prec_geno", "prev_geno", "next_geno","waiting_time", "synon", "mutfixtime","change_in_num_beneneigh", "syn_before", "change_largest_fitness_neigh","num_bene_prec_sy","num_bene_prec_no","num_bene_prev_sy","num_bene_prev_no"])
    waiting = pd.DataFrame(waiting, columns=["synon", "mutfixtime","syn_before", "num_bene_prec_sy","num_bene_prec_no","num_bene_prev_sy","num_bene_prev_no"])
    return waiting

#directory = "../results/adaptive_walk/gb1/no_mistranslation_popsize1e6/"
#synon = waiting.loc[(waiting["syn_before"]==True) & (waiting["synon"]==False) & (waiting["mutfixtime"]>1e4), "change_in_num_beneneigh"]
#nonsy = waiting.loc[(waiting["syn_before"]==False) & (waiting["synon"]==False) & (waiting["mutfixtime"]>1e4), "change_in_num_beneneigh"]
#an.stats.wilcoxon(waiting.loc[waiting["syn_before"]==True, "num_bene_prec_no"], waiting.loc[waiting["syn_before"]==True, "num_bene_prev_no"])

def prepare_plot_change_bene_nonsyn(directory="../results/adaptive_walk/gb1/", protein=500, threshold=0):
    dirs = [f for f in os.listdir(directory) if not "." in f]
    num_nonsyn = []
    for d in dirs:
        if "no_mistranslation" in d or str(protein) in d.split("_")[0]:
            waiting = get_waiting_times_between_fixations(directory+d+"/", threshold=threshold)
            precmean = waiting.loc[waiting["syn_before"]==True, "num_bene_prec_no"].mean()
            precvar = waiting.loc[waiting["syn_before"]==True, "num_bene_prec_no"].var()
            nextmean = waiting.loc[waiting["syn_before"]==True, "num_bene_prev_no"].mean()
            nextvar = waiting.loc[waiting["syn_before"]==True, "num_bene_prev_no"].var()
            num_nonsyn.append([d, float(d[-3:]), precmean, precvar, nextmean, nextvar, waiting.shape[0]])
    return pd.DataFrame(num_nonsyn, columns=["dir", "N", "mean_before", "var_before", "mean_after", "var_after", "n_obs"])

def plot_change_bene_nonsyn(num_nonsy, N=1e6):
    # reformulate
    mean_before = []
    var_before = []
    mean_after = []
    var_after = []
    for row in range(num_nonsy.shape[0]):
        popsize = num_nonsy.loc[row, "N"]
        if popsize == N:
            simtype = num_nonsy.loc[row, "dir"]
            if "no_mistranslation" in simtype:
                sim = "Without mistranslation"
            else:
                sim = "With mistranslation"
            mb = num_nonsy.loc[row, "mean_before"]
            vb = num_nonsy.loc[row, "var_before"]
            ma = num_nonsy.loc[row, "mean_after"]
            va = num_nonsy.loc[row, "mean_after"]
            mean_before.append(mb)
            var_before.append(vb)
            mean_after.append(ma)
            var_after.append(va)
#            data.append([sim, "before", mb, vb])
#            data.append([sim, "after", ma, va])
#    data = pd.DataFrame(data, columns=["simtype", "timing", "mean_nonsy", "var_nonsy"])
#    sns.barplot(x="simtype", y="mean_nonsy", hue="timing", data=data)

    x = np.arange(2)  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(4.8,3.6))
    rects1 = ax.bar(x - width/2, mean_before, width, label='Men')
    rects2 = ax.bar(x + width/2, mean_after, width, label='Women')
    plt.show()

#    synon = []
#    nonsy = []
#    for w,s,n,t,p in zip(wait,syn,nben,mutim,synprec):
#        if p and t>1e4:
#            synon.append(n)
#        elif t > 1e4:
#            nonsy.append(n)
#    plt.hist([synon, nonsy], label=["synon", "nonsy"]); plt.legend(); plt.show()

def get_waiting_times_before_nonsyn_fixations(directory):
    flist = [f for f in os.listdir(directory) if "walk" in f]
    wait = []
    mutim = []
    synprec = [] # Is the current nonsynonymous mutation preceded by a synonymous mutation?
    for f in sorted(flist):
        walk = np.loadtxt(directory+f, dtype=str)
        time = 0
        prec = []
        for i in range(len(walk)-2):
            # do not consider last mutation: the time it exists is only until
            # the end of the simulation and not the next mutation
            prev = walk[i, 0]
            mut = walk[i+1, 0]
            w = int(walk[i, 1])
            time += int(walk[i, 1])
            # This will be the preceding mutation for the next round
            if evo.hamming(evo.translate(prev), evo.translate(mut)) == 0:
                # Do not record synonymous mutations
                prec.append(True)
            elif len(prec) == 0:
                # Ignore. This is the first mutation, not preceded by anything
                # Merely record it was nonsynonymous
                prec.append(False)
            else:
                wait.append(w)
                mutim.append(time)
                synprec.append(prec[-1])
                prec.append(False)
    return wait, synprec, mutim

#directory = "../results/adaptive_walk/pard3e2/"
#listdir = an.os.listdir(directory)
#for f in listdir:
#    if not f.endswith(".pdf") and not f.startswith("."):
#        wait, synprec, mutim = an.get_waiting_times_before_nonsyn_fixations(directory+f+"/")
#        synon = [w for w,s,m in zip(wait, synprec, mutim) if s and m > 1e4]
#        nonsy = [w for w,s,m in zip(wait, synprec, mutim) if not s and m > 1e4]
#        print(f)
#        print("synon: "+str(np.mean(synon)))
#        print("nonsy: "+str(np.mean(nonsy)))



################################################################################
#                  G E P H I    P L O T T I N G                                #
################################################################################

    def plot_neutral_network_fitnesses(network_seq, network_neigh, landscape, seq='ATTAGAGCTTGT'):
    #    seqdata = {}
        dist = []
        ndist = []
        mfit = []
        nmfit = []
        fvar = []
        Ne = []
        for s in network_seq[seq]:
            landscape.execute("SELECT * FROM gb1 WHERE seq = ?", (s,))
            data = landscape.fetchone()
    #        seqdata[data[0]] = [v for v in data[1:]]
            dist.append(evo.hamming(seq, data[0]))
            mfit.append(data[1])
            fvar.append(data[2])
            Ne.append(evo.popsize_reduction(1e6, data[2]/data[1]**2))
        for s in network_neigh [seq]:
            landscape.execute("SELECT * FROM gb1 WHERE seq = ?", (s,))
            data = landscape.fetchone()
            ndist.append(evo.hamming(seq, data[0]))
            nmfit.append(data[1])
        plt.scatter(dist, mfit)
        plt.scatter(ndist, nmfit, color="red", alpha=0.5)
        plt.xlabel("Hamming distance")
        plt.ylabel("Mean fitness")
        plt.show()
        plt.scatter(dist, [v/m**2 for v,m in zip(fvar, mfit)])
        plt.xlabel("Hamming distance")
        plt.ylabel("Fitness noise")
        plt.show()

    def make_neutral_network_with_neigh(network_seq, network_neigh, landscape, seq='ATTAGAGCTTGT', N=1e6):
        # Make a neutral network with all its non-neutral single-step neighbours
        n = nx.Graph()
        landscape.execute("SELECT fmean FROM gb1 WHERE seq = ?", (seq,))
        fmean = landscape.fetchone()[0]
        for s in network_seq[seq]:
            landscape.execute("SELECT * FROM gb1 WHERE seq = ?", (s,))
            data = landscape.fetchone()
            fnoise = data[2]/data[1]**2
            Ne = evo.popsize_reduction(N, fnoise)
            n.add_node(s, mfit=data[1], fvar=data[2], fnoise=fnoise, Ne=Ne,
                viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.5, bytes=True)[:-1])}},
                on_network=True) # Middle color
        for s in network_neigh [seq]:
            landscape.execute("SELECT * FROM gb1 WHERE seq = ?", (s,))
            data = landscape.fetchone()
            fnoise = data[2]/data[1]**2
            Ne = evo.popsize_reduction(N, fnoise)
            if data[1] > fmean: # beneficial
                n.add_node(s, mfit=data[1], fvar=data[2],fnoise=fnoise, Ne=Ne,
                    viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.9999, bytes=True)[:-1])}},
                    on_network=False)
            else:
                n.add_node(s, mfit=data[1], fvar=data[2], fnoise=fnoise, Ne=Ne,
                    viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.0, bytes=True)[:-1])}},
                    on_network=False)
        for node in n.nodes:
            n.nodes[node]["viz"]["color"]["a"] = 0
        for n1,n2 in itertools.combinations(n.nodes, 2):
            if evo.hamming(n1,n2) == 1:
                n.add_edge(n1,n2)
        return n

    def make_neutral_network_with_neigh_compact(network_seq, network_neigh, landscape, seq='ATTAGAGCTTGT', N=1e6):
        # Make a neutral network with all its non-neutral single-step neighbours
        # For those sequences with other fitness than the network, organise according to amino-acid sequences and connect to
        # genotype
        n = nx.Graph()
        landscape.execute("SELECT fmean FROM gb1 WHERE seq = ?", (seq,))
        fmean = landscape.fetchone()[0]
        for s in network_seq[seq]:
            landscape.execute("SELECT * FROM gb1 WHERE seq = ?", (s,))
            data = landscape.fetchone()
    #        fnoise = data[2]/data[1]**2
    #        Ne = evo.popsize_reduction(N, fnoise)
            n.add_node(s, mfit=np.e**data[1], fvar=data[2],
                viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.5, bytes=True)[:-1])},
                    "size": 1},
                on_network=True) # Middle color
        bene = {}
        dele = {}
        for s in network_neigh [seq]:
            landscape.execute("SELECT fmean FROM gb1 WHERE seq = ?", (s,))
            data = landscape.fetchone()[0]
            if data > fmean:
                bene[s] = data
            else:
                dele[s] = data
        n.add_node("beneficial", mfit=np.e**(np.log(np.max([n.nodes[s]["mfit"] for s in n.nodes]))+0.0001), fvar=0,
            viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.9999, bytes=True)[:-1])},
                "size":np.log10(len(bene))},
            on_network=False)
        n.add_node("deleterious", mfit=np.e**(np.log(np.min([n.nodes[s]["mfit"] for s in n.nodes]))-0.0001), fvar=0,
            viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.0, bytes=True)[:-1])},
                "size":np.log10(len(dele))},
            on_network=False)
        for node in n.nodes:
            n.nodes[node]["viz"]["color"]["a"] = 0
        allseqs = [s for s in bene.keys()] + [s for s in dele.keys()] + [s for s in n.nodes if not s in ["beneficial", "deleterious"]]
        edges = []
        for s1,s2 in itertools.combinations(allseqs, 2):
            if evo.hamming(s1,s2) == 1:
                pair = []
                for s in [s1,s2]:
                    if s in n.nodes:
                        pair.append(s)
                    elif s in bene.keys():
                        pair.append("beneficial")
                    elif s in dele.keys():
                        pair.append("deleterious")
                    else:
                        raise RuntimeError("Something is wrong, and I don't know what")
                if pair[0] != pair[1] and not ("beneficial" in pair and "deleterious" in pair): # No need to show connections to itself,
                #nor between bene and dele
                    edges.append(tuple(pair))
        for pair,w in Counter(edges).items():
            if "beneficial" in pair:
                color = {"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.9999, bytes=True)[:-1])}}
            elif "deleterious" in pair:
                color = {"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.0, bytes=True)[:-1])}}
            else:
                color = {"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.5, bytes=True)[:-1])}}
            n.add_edge(pair[0], pair[1], weight=np.log10(w+1), viz=color)
        for edge in n.edges:
            n.edges[edge]["viz"]["color"]["a"] = 1
        # Normalize to maximize differences btwn nodes
        mfits = [n.nodes[s]["mfit"] for s in n.nodes]
        minf = np.min(mfits)
        maxf = np.max(mfits)
        for s in n.nodes:
            f = n.nodes[s]["mfit"]
            fnew = (f-minf) / (maxf - minf)
            n.nodes[s]["mfit"] = fnew
        return n

    def make_neutral_network_with_neigh_polypeptide(network_seq, network_neigh, landscape, gb1, seq='ATTAGAGCTTGT', N=1e6):
        # Make a neutral network with all its non-neutral single-step neighbours
        # For those sequences with other fitness than the network, organise according to amino-acid sequences and connect to
        # genotype
        n = nx.Graph()
        fexpt = gb1[evo.translate(seq)]
        for s in network_seq[seq]:
            landscape.execute("SELECT * FROM gb1 WHERE seq = ?", (s,))
            data = landscape.fetchone()
    #        fnoise = data[2]/data[1]**2
    #        Ne = evo.popsize_reduction(N, fnoise)
            n.add_node(s, mfit=data[1], fvar=data[2],
                viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.5, bytes=True)[:-1])},
                    "size": 1},
                on_network=True) # Middle color
        bene = {}
        dele = {}
        for s in network_neigh[seq]:
            peptide = evo.translate(s)
            fit = gb1[peptide]
            if fit > fexpt:
                if not peptide in bene.keys():
                    bene[peptide] = [fit, 1]
                else:
                    bene[peptide][1] += 1
            else:
                if not peptide in dele.keys():
                    dele[peptide] = [fit, 1]
                else:
                    dele[peptide][1] += 1
        for peptide in bene.keys():
            n.add_node(peptide, mfit=bene[peptide][0], fvar=0,
                viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.9999, bytes=True)[:-1])},
                    "size":np.log10(bene[peptide][1])},
                on_network=False)
        for peptide in dele.keys():
            n.add_node(peptide, mfit=dele[peptide][0], fvar=0,
                viz={"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.0, bytes=True)[:-1])},
                    "size":np.log10(dele[peptide][1])},
                on_network=False)
        for node in n.nodes:
            n.nodes[node]["viz"]["color"]["a"] = 0
        allseqs = network_seq[seq] + network_neigh[seq]
        edges = []
        colors = {}
        for s1,s2 in itertools.combinations(allseqs, 2):
            if evo.hamming(s1,s2) == 1:
                pair = []
                kind = []
                for s in [s1,s2]:
                    if s in n.nodes:
                        pair.append(s)
                        kind.append("neutral")
                    elif evo.translate(s) in bene.keys():
                        pair.append(evo.translate(s))
                        kind.append("beneficial")
                    elif evo.translate(s) in dele.keys():
                        pair.append(evo.translate(s))
                        kind.append("deleterious")
                    else:
                        raise RuntimeError("Something is wrong, and I don't know what")
                if pair[0] != pair[1] and "neutral" in kind: # No need to show connections to itself
                #get rid of all edges that do not involve the focal neutral networks (simplifies)
                    edges.append(tuple(pair))
                    if "beneficial" in kind:
                        colors[tuple(pair)] = {"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.9999, bytes=True)[:-1])}}
                    elif "deleterious" in kind:
                        colors[tuple(pair)] = {"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.0, bytes=True)[:-1])}}
                    else:
                        colors[tuple(pair)] = {"color":{k:str(v) for k,v in zip(["r","g","b"], wes(0.5, bytes=True)[:-1])}}
        for pair,w in Counter(edges).items():
            n.add_edge(pair[0], pair[1], weight=np.log10(w+1), viz=colors[pair])
        for edge in n.edges:
            n.edges[edge]["viz"]["color"]["a"] = 1
        # Normalize to maximize differences btwn nodes in network
        mfits = [n.nodes[s]["mfit"] for s in n.nodes if len(s) > 4]
        medianf = np.median(mfits)
        for s in n.nodes:
            f = n.nodes[s]["mfit"]
            n.nodes[s]["mfit[Z]"] = 1 / (1 + np.exp(-1000* (f - medianf )))
        # Rank node fitnesses (0 -> lowest fitness)
        # Use lists to conserve order
    #    seqs = [s for s in n.nodes]
    #    mfits = [n.nodes[s]["mfit"] for s in seqs]
    #    index = np.unique(mfits, return_inverse = True)[1]
    #    for s,i in zip(seqs,index):
    #        n.nodes[s]["rank[Z]"] = i
        return n

    def prepare_pard3_gephi(pard3e2, pard3e3):
        n = nx.Graph()
        for k in pard3e2.keys():
            e2fit = pard3e2[k]
            e3fit = pard3e3[k]
            # determine the color based on the ration between fitnesses
            if e3fit == 0 and e2fit == 0:
                skew = 0.5
            elif e3fit == 0:
                skew = 0.9999
            elif e2fit == 0:
                skew = 0
            else:
                x = np.log10(e2fit/e3fit)
                skew = 1 / ( 1 + np.exp(-5 * (x) ) )
            if skew == 1:
                skew = 0.9999
            n.add_node(k, fitness = max(e2fit, e3fit), skew=skew,
                    viz={"color": {k:str(v) for k,v in zip(["r", "g", "b"], wes(skew, bytes=True)[:-1])},
                        "size": 1},
                    on_network=True)
        # Set the alpha channel to intransparent
        for node in n.nodes:
            n.nodes[node]["viz"]["color"]["a"] = 0
            n.nodes[node]["fitness[Z]"] = n.nodes[node]["fitness"]
        # Make edges
        allseqs = [k for k in pard3e2.keys()]
        edges = []
        for s1,s2 in itertools.combinations(allseqs, 2):
            if evo.hamming(s1, s2) == 1:
                n.add_edge(s1, s2)
        return n

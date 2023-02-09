


import os
import itertools
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import evolution as evo
import mistranslation_rate_estimate as mre
import sqlite3 as sql
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mplcolors
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import Counter, OrderedDict
from scipy import stats, interpolate
from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess
from  scikit_posthocs import posthoc_nemenyi_friedman as nemenyi
from  scikit_posthocs import posthoc_dunn as dunn
from neutral_networks import find_neutral_neighbours
import pickle
import palettable as pt

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


def plot_figure_2():
    # get data
    landscape = evo.read_fitness_dictionary("../data/gb1_fitdict.txt")
    directory = "../results/selection_power/gb1/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    gs = gridspec.GridSpec(6, 12)
    fig = plt.figure(figsize=(12.6,6.3), constrained_layout=False)
    # panel A
    fitch = results["fit_wt_mis"].values - results["fit_wt"].values
    ax_main = fig.add_subplot(gs[1:6, :5])
    ax_xDist = fig.add_subplot(gs[0, :5],sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:6, 5:6],sharey=ax_main)
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
    ax_xDist.annotate("A", xy=(-0.3,1.2), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel B
    # get data
    nresults = pd.read_csv(directory+"corrphenogeno.csv")
    panb = fig.add_subplot(gs[0:3, 8:])
    panb.plot(nresults["fit_neigh_nt"], nresults["fit_neigh_aa"], ".",
    color="grey", mec='none', alpha=0.4)
    panb.set_xscale("log")
    panb.set_yscale("log")
    panb.set_xlabel("Mean fitness of \n genetic neighbours", fontsize=16)
    panb.set_ylabel("Mean fitness of \n phenotypic neighbours", fontsize=16)
    panb.tick_params(axis='both', labelsize=14)
    panb.annotate("B", xy=(-0.5,1.06), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel C
    exprlevel = [1,10,100,500,1000]
    panc = fig.add_subplot(gs[4:, 8:])
    ne = []
    for i in exprlevel:
        results = pd.read_csv("../results/selection_power/gb1/proteins{}_popsize1e6/results.csv".format(i))
        for v in results["Ne"].to_list():
            ne.append([i, v])
    ne = pd.DataFrame(ne, columns=["pr", "ne"])
    sns.violinplot(x="pr", y="ne", data=ne, ax=panc,
        inner=None, color="grey", scale="width")
    sc = panc.errorbar(x=range(len(exprlevel)),
        y=[ne.loc[ne["pr"]==pr, "ne"].median() for pr in exprlevel],
        yerr=[ne.loc[ne["pr"]==pr, "ne"].std() for pr in exprlevel],
        linewidth=0, elinewidth=2, marker="d", color="white", capsize=0)
    sc.lines[0].set_zorder(12)
    sc.lines[0].set_zorder(12)
    panc.set_ylim(bottom=-5e3,top=1e6*1.01)
    panc.set_xlabel("Protein expression level (per cell)", fontsize=16)
    panc.set_ylabel("Effective\npopulation "+r"size ($N_e$)", fontsize=16)
    panc.ticklabel_format(style="sci", scilimits=(0,0), axis="y", useMathText=True, useOffset=False)
    panc.tick_params(axis='both', which='major', labelsize=14)
    panc.yaxis.offsetText.set_fontsize(14)
    panc.annotate("C", xy=(-0.5,1.4), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
#    plt.tight_layout()
    fig.savefig("../writing/figures/figure_2.png")
    plt.close()

def plot_figure_3A():
    directory = "../results/selection_power/gb1/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    save_figure = "../writing/figures/figure_3A.png"
    # Panel A
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
    fig.update_layout(margin=dict(l=60, r=60, t=60, b=60), width=1209.6, height=450)

    # add annotation
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0,
                                        y=1.12,
                                        showarrow=False,
                                        text="Without mistranslation",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=1.,
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
                                        x=0.48,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Neutral",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0.87,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Deleterious",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black', size=32),
                                        x=-0.05,
                                        y=1.15,
                                        showarrow=False,
                                        text="<b>A</b>",
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
    x0=0.40, y0=-0.06, x1=0.47, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.5)))
    fig.add_shape(type="rect",
    x0=0.79, y0=-0.06, x1=0.86, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.0)))
    fig.write_image(save_figure)

def plot_figure_3rest():
    directory = "../results/selection_power/gb1/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    fig = plt.figure(figsize=(12.6,9.5))
    # Panel B
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
    axB = plt.subplot2grid((2,2), (0,0), colspan=1)
    axB = sns.histplot(x="difference", hue="kind", data=data, multiple="dodge",
                log_scale=10, palette=[mplcolors.rgb2hex(greenbrown(0.0)),
                mplcolors.rgb2hex(greenbrown(0.9999))], alpha=1,
                edgecolor='none', ax=axB)
#    plt.hist(syn, bins=slogbins, label="synonymous")
    axB.set_ylabel("Number of observations", fontsize=16)
    axB.set_xlabel("Absolute difference in fitness", fontsize=16)
    axB.tick_params(axis="both", labelsize=14)
    axB.get_legend().set_title(None)
    plt.setp(axB.get_legend().get_texts(), fontsize=14)
    axB.annotate("B", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel C
    axC = plt.subplot2grid((2,2), (0,1), colspan=1)
    directory="../results/selection_power/gb1/"
    perc_neut = {}
    for c,n in enumerate([4, 6, 8]):
        perc_neut[n] = []
        for p in [1, 10, 100]:
            results = pd.read_csv(directory+"proteins{}_popsize1e{}/results.csv".format(p,n))
            results = classify_results(results)
            perc = sum([v=="neutral" for v in results["class_mis"]]) / results.shape[0]
            perc_neut[n].append(perc * 100)
        axC.plot([1, 10, 100], perc_neut[n], "-o", label = r"10$^{}$".format(n),
            color = redscale(c/2))
    axC.set_xlabel("Number of proteins per cell", fontsize=16)
    axC.set_ylabel("Percentage neutral \nwith mistranslation", fontsize=16)
    axC.set_xscale("log")
    axC.set_yscale("log")
    legend = axC.legend(loc="center right", fontsize=14, frameon=False,
        bbox_to_anchor=(1.,0.55))
    legend.set_title('Population size',prop={'size':14})
    axC.tick_params(axis="both", labelsize=14)
    axC.annotate("C", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel D
    directory = "../results/selection_power/gb1/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    axD = plt.subplot2grid((2,2), (1,0), colspan=1)
    if not "class" in results.columns:
        results = classify_results(results)
    N = results.loc[0, "N"]
    benidx = (results["class"] == results["class_mis"]) & (results["class"] == "beneficial")
    bendiff = results.loc[benidx, "fixprob_mis"] - results.loc[benidx, "fixprob"]
    axD.hist(bendiff, bins=100, color=wes(0.99), align="mid")
    axD.set_yscale("log")
    axD.set_ylabel("Frequency", fontsize=16)
    axD.set_xlabel("Change in fixation probability\nof beneficial mutations", fontsize=16)
    axD.locator_params(axis='x', nbins=6)
    axD.tick_params(axis="both", labelsize=14)
    axD.annotate("D", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel E
    Ne = 1e6
    pr = 1
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
    axE = plt.subplot2grid((2,2), (1,1), colspan=1)
    axE.bar(landscapes, num_increase, color="grey")
    axE.set_ylabel("% beneficial mutations with \n increased fixation probability", fontsize=16)
    axE.set_xticks(ticks=[0,1,2], labels=landscape_labels, fontsize=16)
    axE.tick_params(axis="both", labelsize=14)
    axE.annotate("E", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    plt.tight_layout()
    fig.savefig("../writing/figures/figure_3_rest.png")
    plt.close()

def plot_figure_4():
    directory="../results/peak_counting/"
    N="1e8"
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

    fig = plt.figure(figsize=(12.6, 4.7))
    # Panel A
    axA = plt.subplot2grid((1,2), (0,0), colspan=1)
    axA = sns.barplot(x="landscape", y="Number of fitness peaks", hue="mistr", hue_order=order,
            data=peak, palette=[mplcolors.rgb2hex(wes(0.)),
            mplcolors.rgb2hex(bluescale(0.1)), mplcolors.rgb2hex(bluescale(1.0))],
            ax = axA)
#    ax.get_legend().set_title(None)
    axA.legend([],[], frameon=False)
    axA.set_xlabel("")
    axA.set_ylabel(axA.get_ylabel(), fontsize=16)
    axA.tick_params(axis="both", labelsize=14)
    axA.annotate("A", xy=(0,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)

    # Panel B
    directory="../results/epistasis/"
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
    labels = []
    magnitude = []
    reciprocal = []
    signepist = []
    for row in epiprop:
        labels.append(", \n".join([row[0], row[1]]))
        reciprocal.append(row[2])
        signepist.append(row[3])
        magnitude.append(row[4])

    axB = plt.subplot2grid((1,2), (0,1), colspan=1)
    colors = [mplcolors.rgb2hex(wes(0.)), mplcolors.rgb2hex(bluescale(0.1)), mplcolors.rgb2hex(bluescale(1.0))]
    axB.bar(labels, magnitude, label="magnitude", color=colors*3)
    axB.bar(labels, signepist, label="simple sign", color=colors*3)
    axB.bar(labels, reciprocal, label="reciprocal",color=colors*3)

    h,l = axB.get_legend_handles_labels() # get the handles we want to modify
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
    axB.set_xticks(np.arange(1, 9, 3))
    axB.set_xticklabels(landscape_labels, rotation = 0, fontsize=14)
    axB.tick_params(axis='y', which='major', labelsize=14)

    n=[]
    for i,H in enumerate([None, "/"*3, "x"*3]):
        n.append(axB.bar(0, 0, color="white", hatch=H, edgecolor="k"))
    pr=[]
    names = ["without mistranslation", "mistranslation, 500 proteins", "mistranslation, 1 protein"]
    for name,c in zip(names, colors):
        pr.append(axB.bar(0,0, color=c, label=name))

    l1 = axB.legend(pr, names, bbox_to_anchor=(0., 1.), loc="lower left", frameon=False,fontsize=14)
    l2 = plt.legend(n, ["magnitude", "simple sign", "reciprocal"], bbox_to_anchor=(0.65, 1.), loc="lower left", frameon=False,fontsize=14)
    axB.add_artist(l1)

    axB.set_ylabel("Fraction of samples (%)", fontsize=16)
    axB.annotate("B", xy=(0,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)

    plt.tight_layout()
    fig.savefig("../writing/figures/figure_4.png")
    plt.close()

def plot_figure_5():
    directory = "../results/adaptive_walk/gb1/proteins500_popsize1e8/"
    nomisdir = "../results/adaptive_walk/gb1/no_mistranslation_popsize1e8/"
    numfalse = get_mutnum_to_endpoint(nomisdir)
    numtrue =  get_mutnum_to_endpoint(directory)
    gs = gridspec.GridSpec(8, 8)
    fig = plt.figure(figsize=(12.6, 9.5), constrained_layout=False)
    # Panel A
    axA = fig.add_subplot(gs[:3, :3])
    axA.hist([numfalse, numtrue], density=False, label=["without mistranslation", "with mistranslation"],
                color=[wes(0.9999), wes(0)])
    axA.set_xlabel("Number of fixation events", fontsize=16)
    axA.set_ylabel("Frequency", fontsize=16)
    axA.tick_params(axis='both', which='major', labelsize=14)
    axA.legend(fontsize=14, frameon=False, bbox_to_anchor=(0., 1.), loc="lower left",)
    axA.annotate("A", xy=(0,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)
    # Panel B
    # Read data and prepare
    chfit, mutnum, synon = get_fitness_change_time(directory)
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

    axB1 = fig.add_subplot(gs[:4, 4:8])
    axB2 = fig.add_subplot(gs[4:8, 4:8], sharex=axB1)
    axB1.scatter(non[:,0], non[:,1], alpha=0.2, edgecolor='none', marker='.', color=greenbrown(0.9999))
    axB2.scatter(syn[:,0], syn[:,1], alpha=0.2, edgecolor='none', marker='.', color=greenbrown(0.))
    # Inset in B1: first 1000 mutations
    begi = non[non[:,0] <= 1000, :]
    inax = inset_axes(axB1, width="100%", height="100%", loc="upper right", bbox_to_anchor=(0.4,1-0.42,.55,.4), bbox_transform=axB1.transAxes)
    inax.scatter(begi[:,0], begi[:,1], alpha=0.2, edgecolor='none', marker='.', color=greenbrown(0.9999) )
    inax.set_xlabel("Number of mutations", fontsize=16)
    inax.set_ylabel("Fitness change\nupon fixation", fontsize=16)
    inax.set_yticks([0, 4, 8])
    inax.tick_params(axis="both", labelsize=14)
    inax.set_xlim(0,1000)
    # Legend
    legend_elem = [Line2D([0], [0], marker='.', color='w', label='Nonsynonymous',
                          markerfacecolor=greenbrown(0.9999), markersize=12),
                    Line2D([0], [0], marker='.', color='w', label='Synonymous',
                         markerfacecolor=greenbrown(0.), markersize=12),
                          ]
    axB2.legend(handles=legend_elem, frameon=False, fontsize=14)
    axB2.set_xlabel("Number of mutations", fontsize=16)
    axB2.set_xlim(-500,1e5+500)
    plt.setp(axB1.get_xticklabels(), visible=False)
    axB1.tick_params(axis='both', which='major', labelsize=14)
    axB2.tick_params(axis='both', which='major', labelsize=14)
    fig.text(0.45, 0.5, 'Fitness change upon fixation', va='center', rotation='vertical', size=16)
    axB1.annotate("B", xy=(0,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)

    # Panel C
    nomisprops = get_misrates_over_time(nomisdir)
    misprops = get_misrates_over_time(directory)
    # mistranslation rates of the landscape
    misrates = np.loadtxt("../data/gb1_mistranslation_rates.txt")
    proptime = get_percent_misrates_over_time(nomisprops, misprops, np.mean(misrates))

    axC = fig.add_subplot(gs[4:, :3])
    for mis, vals in proptime.items():
        x_pos, means, stderr = vals
        if mis.startswith("without"):
            c = wes(0.9999)
        else:
            c = wes(0.0)
        axC.fill_between(x_pos, means-1.96*stderr, means+1.96*stderr, alpha=0.25,
            color=c)
        axC.plot(x_pos, means, color=c, label=mis)
    axC.set_xlabel("Number of mutations", fontsize=16)
#    axC.set_ylabel("Deviation from mean\n"+r"mistranslation rate ($\sigma$)", fontsize=16)
    axC.set_ylabel("Percentage of mean\n"+r"mistranslation rate", fontsize=16)
    axC.set_xticks(np.linspace(0, max(x_pos), 5))
    axC.tick_params(axis='both', which='major', labelsize=14)
    axC.set_xlim(0,max(x_pos))
    axC.set_ylim(98.5,100.5)
    axC.annotate("C", xy=(0,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)

    fig.savefig("../writing/figures/figure_5.png")
    plt.close()

def plot_figure_SI_1():
    nobs_threshold=18
    misrates = mre.get_rates()
    nobs, irmed, err = mre.get_summary(misrates)
    res = mre.regression(nobs, np.log10(irmed), nobs_threshold=0)
    respart = mre.regression(nobs, np.log10(irmed), nobs_threshold=nobs_threshold)
    # Panel A
    fig = plt.figure(figsize=(12.6, 4.7))
    axA = plt.subplot2grid((1,2), (0,0), colspan=1)
    axA.plot(nobs, irmed, '+', color="black")
    axA.plot(np.array(range(max(nobs)+1)), 10**(res.intercept + res.slope*np.array(range(max(nobs)+1))), color=wes(0.9999), label="no threshold")
    axA.plot(np.array(range(max(nobs)+1)), 10**(respart.intercept + respart.slope*np.array(range(max(nobs)+1))), color=wes(0.), label="with threshold")
    axA.errorbar(nobs, irmed, yerr=err, linestyle="", color="grey")
    axA.set_yscale("log")
    axA.set_xlabel("Number of samples", fontsize=14)
    axA.set_ylabel("Mistranslation rate estimate", fontsize=14)
    axA.tick_params(axis='both', which='major', labelsize=14)
    axA.legend(fontsize=14, frameon=False)
    axA.annotate("A", xy=(-0.2,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)

    # Panel B
    axB = plt.subplot2grid((1,2), (0,1), colspan=1)
    minrates = []
    excluded = []
    for i in range(51):
        res = mre.regression(nobs, np.log10(irmed), i)
        minrates.append(10**res.intercept)
        excluded.append(sum(nobs < i)/len(nobs)*100)

    color = "black"
    axB.plot(range(51), np.log10(minrates), color="black")
    axB.set_xlabel("Threshold for exclusion\n(minimum number of samples)", size=14)
    axB.set_ylabel("Minimum mistranslation rate" "\n" r"log$_{10}$(regression intercept)", color=color, size=14)
    axB.tick_params(axis='y', labelcolor=color, labelsize=14)
    axB.tick_params(axis='x', labelcolor=color, labelsize=14)
    axB.set_xlim(0,50)
    axB2 = axB.twinx()
    color = wes(0)
    axB2.set_ylabel("Percentage of data excluded", color=color, size=14)  # we already handled the x-label with ax1
    axB2.plot(range(51), excluded, color=color)
    axB2.tick_params(axis='y', labelcolor=color, labelsize=14)
    axB2.set_ylim(0,100)
    axB.annotate("B", xy=(-0.2,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)

    plt.tight_layout()
    fig.savefig("../writing/figures/figure_SI_1.png")
    plt.close()

def plot_figure_SI_2AC():
    landscape = evo.read_fitness_dictionary("../data/pard3e3.txt")
    directory = "../results/selection_power/pard3e3/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    gs = gridspec.GridSpec(6, 12)
    fig = plt.figure(figsize=(12.6,6.3), constrained_layout=False)
    # panel A
    fitch = results["fit_wt_mis"].values - results["fit_wt"].values
    ax_main = fig.add_subplot(gs[1:6, :5])
    ax_xDist = fig.add_subplot(gs[0, :5],sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:6, 5:6],sharey=ax_main)
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
    ax_xDist.annotate("A", xy=(-0.3,1.2), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel B
    # get data
    nresults = pd.read_csv(directory+"corrphenogeno.csv")
    panb = fig.add_subplot(gs[0:3, 8:])
    panb.plot(nresults["fit_neigh_nt"], nresults["fit_neigh_aa"], ".",
    color="grey", mec='none', alpha=0.4)
    panb.set_xscale("log")
    panb.set_yscale("log")
    panb.set_xlabel("Mean fitness of \n genetic neighbours", fontsize=16)
    panb.set_ylabel("Mean fitness of \n phenotypic neighbours", fontsize=16)
    panb.tick_params(axis='both', labelsize=14)
    panb.annotate("B", xy=(-0.5,1.06), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel C
    exprlevel = [1,500]
    panc = fig.add_subplot(gs[4:, 8:])
    ne = []
    for i in exprlevel:
        results = pd.read_csv("../results/selection_power/gb1/proteins{}_popsize1e6/results.csv".format(i))
        for v in results["Ne"].to_list():
            ne.append([i, v])
    ne = pd.DataFrame(ne, columns=["pr", "ne"])
    sns.violinplot(x="pr", y="ne", data=ne, ax=panc,
        inner=None, color="grey", scale="width")
    sc = panc.errorbar(x=range(len(exprlevel)),
        y=[ne.loc[ne["pr"]==pr, "ne"].median() for pr in exprlevel],
        yerr=[ne.loc[ne["pr"]==pr, "ne"].std() for pr in exprlevel],
        linewidth=0, elinewidth=2, marker="d", color="white", capsize=0)
    sc.lines[0].set_zorder(12)
    sc.lines[0].set_zorder(12)
    panc.set_ylim(bottom=-5e3,top=1e6*1.01)
    panc.set_xlabel("Protein expression level (per cell)", fontsize=16)
    panc.set_ylabel("Effective\npopulation "+r"size ($N_e$)", fontsize=16)
    panc.ticklabel_format(style="sci", scilimits=(0,0), axis="y", useMathText=True, useOffset=False)
    panc.tick_params(axis='both', which='major', labelsize=14)
    panc.yaxis.offsetText.set_fontsize(14)
    panc.annotate("C", xy=(-0.5,1.4), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
#    plt.tight_layout()
    fig.savefig("../writing/figures/figure_SI_2AC.png")
    plt.close()

def plot_figure_SI_2D():
    directory = "../results/selection_power/pard3e3/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    save_figure = "../writing/figures/figure_SI_2D.png"
    # Panel A
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
    fig.update_layout(margin=dict(l=60, r=60, t=60, b=60), width=1209.6, height=450)

    # add annotation
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0,
                                        y=1.12,
                                        showarrow=False,
                                        text="Without mistranslation",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=1.,
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
                                        x=0.48,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Neutral",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0.87,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Deleterious",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black', size=32),
                                        x=-0.05,
                                        y=1.15,
                                        showarrow=False,
                                        text="<b>D</b>",
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
    x0=0.40, y0=-0.06, x1=0.47, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.5)))
    fig.add_shape(type="rect",
    x0=0.79, y0=-0.06, x1=0.86, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.0)))
    fig.write_image(save_figure)
    plt.close()

def plot_figure_SI_2EG():
    directory = "../results/selection_power/pard3e3/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    fig = plt.figure(figsize=(12.6,4.7))
    # Panel E
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
    axE = plt.subplot2grid((1,3), (0,0), colspan=1)
    axE = sns.histplot(x="difference", hue="kind", data=data, multiple="dodge",
                log_scale=10, palette=[mplcolors.rgb2hex(greenbrown(0.0)),
                mplcolors.rgb2hex(greenbrown(0.9999))], alpha=1,
                edgecolor='none', ax=axE)
#    plt.hist(syn, bins=slogbins, label="synonymous")
    axE.set_ylabel("Number of observations", fontsize=16)
    axE.set_xlabel("Absolute difference in fitness", fontsize=16)
    axE.tick_params(axis="both", labelsize=14)
    axE.get_legend().set_title(None)
    plt.setp(axE.get_legend().get_texts(), fontsize=14)
    axE.annotate("E", xy=(-0.3,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel F
    axF = plt.subplot2grid((1,3), (0,1), colspan=1)
    directory="../results/selection_power/pard3e3/"
    perc_neut = {}
    for c,n in enumerate([4, 6, 8]):
        perc_neut[n] = []
        for p in [1, 500]:
            results = pd.read_csv(directory+"proteins{}_popsize1e{}/results.csv".format(p,n))
            results = classify_results(results)
            perc = sum([v=="neutral" for v in results["class_mis"]]) / results.shape[0]
            perc_neut[n].append(perc * 100)
        axF.plot([1, 500], perc_neut[n], "-o", label = r"10$^{}$".format(n),
            color = redscale(c/2))
    axF.set_xlabel("Number of proteins per cell", fontsize=16)
    axF.set_ylabel("Percentage neutral \nwith mistranslation", fontsize=16)
    axF.set_xscale("log")
    axF.set_yscale("log")
    legend = axF.legend(loc="center right", fontsize=14, frameon=False,
        bbox_to_anchor=(1.,0.55))
    legend.set_title('Population size',prop={'size':14})
    axF.tick_params(axis="both", labelsize=14)
    axF.annotate("F", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel G
    directory = "../results/selection_power/pard3e3/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    axG = plt.subplot2grid((1,3), (0,2), colspan=1)
    if not "class" in results.columns:
        results = classify_results(results)
    N = results.loc[0, "N"]
    benidx = (results["class"] == results["class_mis"]) & (results["class"] == "beneficial")
    bendiff = results.loc[benidx, "fixprob_mis"] - results.loc[benidx, "fixprob"]
    axG.hist(bendiff, bins=100, color=wes(0.99), align="mid")
    axG.set_yscale("log")
    axG.set_ylabel("Frequency", fontsize=16)
    axG.set_xlabel("Change in fixation probability\nof beneficial mutations", fontsize=16)
    axG.locator_params(axis='x', nbins=6)
    axG.tick_params(axis="both", labelsize=14)
    axG.annotate("G", xy=(-0.3,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)

    plt.tight_layout()
    fig.savefig("../writing/figures/figure_SI_2EG.png")
    plt.close()

def plot_figure_SI_3AC():
    landscape = evo.read_fitness_dictionary("../data/pard3e2.txt")
    directory = "../results/selection_power/pard3e2/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    gs = gridspec.GridSpec(6, 12)
    fig = plt.figure(figsize=(12.6,6.3), constrained_layout=False)
    # panel A
    fitch = results["fit_wt_mis"].values - results["fit_wt"].values
    ax_main = fig.add_subplot(gs[1:6, :5])
    ax_xDist = fig.add_subplot(gs[0, :5],sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:6, 5:6],sharey=ax_main)
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
    ax_xDist.annotate("A", xy=(-0.3,1.2), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel B
    # get data
    nresults = pd.read_csv(directory+"corrphenogeno.csv")
    panb = fig.add_subplot(gs[0:3, 8:])
    panb.plot(nresults["fit_neigh_nt"], nresults["fit_neigh_aa"], ".",
    color="grey", mec='none', alpha=0.4)
    panb.set_xscale("log")
    panb.set_yscale("log")
    panb.set_xlabel("Mean fitness of \n genetic neighbours", fontsize=16)
    panb.set_ylabel("Mean fitness of \n phenotypic neighbours", fontsize=16)
    panb.tick_params(axis='both', labelsize=14)
    panb.annotate("B", xy=(-0.5,1.06), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel C
    exprlevel = [1,500]
    panc = fig.add_subplot(gs[4:, 8:])
    ne = []
    for i in exprlevel:
        results = pd.read_csv("../results/selection_power/pard3e2/proteins{}_popsize1e6/results.csv".format(i))
        for v in results["Ne"].to_list():
            ne.append([i, v])
    ne = pd.DataFrame(ne, columns=["pr", "ne"])
    sns.violinplot(x="pr", y="ne", data=ne, ax=panc,
        inner=None, color="grey", scale="width")
    sc = panc.errorbar(x=range(len(exprlevel)),
        y=[ne.loc[ne["pr"]==pr, "ne"].median() for pr in exprlevel],
        yerr=[ne.loc[ne["pr"]==pr, "ne"].std() for pr in exprlevel],
        linewidth=0, elinewidth=2, marker="d", color="white", capsize=0)
    sc.lines[0].set_zorder(12)
    sc.lines[0].set_zorder(12)
    panc.set_ylim(bottom=-5e3,top=1e6*1.01)
    panc.set_xlabel("Protein expression level (per cell)", fontsize=16)
    panc.set_ylabel("Effective\npopulation "+r"size ($N_e$)", fontsize=16)
    panc.ticklabel_format(style="sci", scilimits=(0,0), axis="y", useMathText=True, useOffset=False)
    panc.tick_params(axis='both', which='major', labelsize=14)
    panc.yaxis.offsetText.set_fontsize(14)
    panc.annotate("C", xy=(-0.5,1.4), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
#    plt.tight_layout()
    fig.savefig("../writing/figures/figure_SI_3AC.png")
    plt.close()

def plot_figure_SI_3D():
    directory = "../results/selection_power/pard3e2/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    save_figure = "../writing/figures/figure_SI_3D.png"
    # Panel A
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
    fig.update_layout(margin=dict(l=60, r=60, t=60, b=60), width=1209.6, height=450)

    # add annotation
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0,
                                        y=1.12,
                                        showarrow=False,
                                        text="Without mistranslation",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=1.,
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
                                        x=0.48,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Neutral",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black',size=24),
                                        x=0.87,
                                        y=-0.14,
                                        showarrow=False,
                                        text="Deleterious",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    fig.add_annotation(dict(font=dict(color='black', size=32),
                                        x=-0.05,
                                        y=1.15,
                                        showarrow=False,
                                        text="<b>D</b>",
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
    x0=0.40, y0=-0.06, x1=0.47, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.5)))
    fig.add_shape(type="rect",
    x0=0.79, y0=-0.06, x1=0.86, y1=-0.12,
    line=dict(color="black", width=0.5),
    fillcolor=mplcolors.rgb2hex(wes(0.0)))
    fig.write_image(save_figure)
    plt.close()

def plot_figure_SI_3EG():
    directory = "../results/selection_power/pard3e2/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    fig = plt.figure(figsize=(12.6,4.7))
    # Panel E
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
    axE = plt.subplot2grid((1,3), (0,0), colspan=1)
    axE = sns.histplot(x="difference", hue="kind", data=data, multiple="dodge",
                log_scale=10, palette=[mplcolors.rgb2hex(greenbrown(0.0)),
                mplcolors.rgb2hex(greenbrown(0.9999))], alpha=1,
                edgecolor='none', ax=axE)
#    plt.hist(syn, bins=slogbins, label="synonymous")
    axE.set_ylabel("Number of observations", fontsize=16)
    axE.set_xlabel("Absolute difference in fitness", fontsize=16)
    axE.tick_params(axis="both", labelsize=14)
    axE.get_legend().set_title(None)
    plt.setp(axE.get_legend().get_texts(), fontsize=14)
    axE.annotate("E", xy=(-0.3,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel F
    axF = plt.subplot2grid((1,3), (0,1), colspan=1)
    directory="../results/selection_power/pard3e2/"
    perc_neut = {}
    for c,n in enumerate([4, 6, 8]):
        perc_neut[n] = []
        for p in [1, 500]:
            results = pd.read_csv(directory+"proteins{}_popsize1e{}/results.csv".format(p,n))
            results = classify_results(results)
            perc = sum([v=="neutral" for v in results["class_mis"]]) / results.shape[0]
            perc_neut[n].append(perc * 100)
        axF.plot([1, 500], perc_neut[n], "-o", label = r"10$^{}$".format(n),
            color = redscale(c/2))
    axF.set_xlabel("Number of proteins per cell", fontsize=16)
    axF.set_ylabel("Percentage neutral \nwith mistranslation", fontsize=16)
    axF.set_xscale("log")
    axF.set_yscale("log")
    legend = axF.legend(loc="center right", fontsize=14, frameon=False,
        bbox_to_anchor=(1.,0.55))
    legend.set_title('Population size',prop={'size':14})
    axF.tick_params(axis="both", labelsize=14)
    axF.annotate("F", xy=(-0.2,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)
    # Panel G
    directory = "../results/selection_power/pard3e2/proteins1_popsize1e6/"
    results = pd.read_csv(directory+"results.csv")
    axG = plt.subplot2grid((1,3), (0,2), colspan=1)
    if not "class" in results.columns:
        results = classify_results(results)
    N = results.loc[0, "N"]
    benidx = (results["class"] == results["class_mis"]) & (results["class"] == "beneficial")
    bendiff = results.loc[benidx, "fixprob_mis"] - results.loc[benidx, "fixprob"]
    axG.hist(bendiff, bins=100, color=wes(0.99), align="mid")
    axG.set_yscale("log")
    axG.set_ylabel("Frequency", fontsize=16)
    axG.set_xlabel("Change in fixation probability\nof beneficial mutations", fontsize=16)
    axG.locator_params(axis='x', nbins=6)
    axG.tick_params(axis="both", labelsize=14)
    axG.annotate("G", xy=(-0.3,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)

    plt.tight_layout()
    fig.savefig("../writing/figures/figure_SI_3EG.png")
    plt.close()

def plot_figure_SI_5():
    nearly_neutral = pd.read_csv("../results/neutral_networks/nearly_neutral_size.csv")
    pr = 1
    fig = plt.figure(figsize=(12.6, 4.7))
    # Panel A
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

    axA = plt.subplot2grid((1,2), (0,0), colspan=1)
    sns.boxplot(x="landscape2", y="logSize", hue="kind", data=subset,
            hue_order=order, palette=[mplcolors.rgb2hex(wes(0.0)),
            mplcolors.rgb2hex(wes(0.9999))],
            saturation=1,
            showmeans=True,
            meanprops={"marker":"o",
                   "markerfacecolor":"white",
                   "markeredgecolor":"black",
                  "markersize":"10"},
            ax = axA)
    axA.set_ylabel(r"Network size (log$_{10}$)", fontsize=16)
    axA.set_xlabel("")
    axA.tick_params(axis='y', labelsize=14)
    axA.tick_params(axis='x', labelsize=14)
    axA.get_legend().set_title(None)
    axA.legend(fontsize=13, frameon=False, loc="upper right")
    axA.annotate("A", xy=(-0.15,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)

    # Panel B
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

    axB = plt.subplot2grid((1,2), (0,1), colspan=1)
    sns.boxplot(x="kind", y="logSize", data=subset, order=["no mistranslation",
        "Ne", "smooth", "mistranslation"], color="grey",
        saturation=1,
        showmeans=True,
        meanprops={"marker":"o",
           "markerfacecolor":"white",
           "markeredgecolor":"black",
           "markersize":"10"},
        ax = axB)
    axB.set_ylabel(r"Network size (log$_{10}$)", fontsize=16)
    axB.tick_params(axis="y", labelsize=14)
    axB.set_xticklabels(["no mistranslation", "drift only", "fitness only", "mistranslation"], fontsize=14)
    axB.set_xlabel("")
    axB.annotate("B", xy=(-0.15,1.), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", weight="bold", fontsize=25)

    plt.tight_layout()
    fig.savefig("../writing/figures/figure_SI_5.png")
    plt.close()

def plot_figure_SI_6():
    # Panel A
    nomisdir = "../results/adaptive_walk/pard3e3/no_mistranslation_popsize1e4/"
    misdir = "../results/adaptive_walk/pard3e3/proteins500_popsize1e4/"
    nomisprops = get_misrates_over_time(nomisdir)
    misprops = get_misrates_over_time(misdir)
    # mistranslation rates of the landscape
    misrates = np.loadtxt("../data/pard3e3_mistranslation_rates.txt")
    proptime = get_percent_misrates_over_time(nomisprops, misprops, np.mean(misrates))

    fig = plt.figure(figsize=(12.6, 4.7))
    # Panel A
    axA = plt.subplot2grid((1,2), (0,0), colspan=1)
    for mis, vals in proptime.items():
        x_pos, means, stderr = vals
        if mis.startswith("without"):
            c = wes(0.9999)
        else:
            c = wes(0.0)
        axA.fill_between(x_pos, means-1.96*stderr, means+1.96*stderr, alpha=0.25,
            color=c)
        axA.plot(x_pos, means, color=c, label=mis)
    axA.set_xlabel("Number of mutations", fontsize=16)
    axA.set_ylabel("Percentage of mean\n"+r"mistranslation rate", fontsize=16)
    axA.set_xticks(np.linspace(0, max(x_pos), 5))
    axA.tick_params(axis='both', which='major', labelsize=14)
    axA.set_xlim(0,max(x_pos))
    axA.set_ylim(98.0,100.5)
    axA.legend(frameon=False, fontsize=14)
    axA.annotate("A", xy=(0,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)
    # Panel A
    nomisdir = "../results/adaptive_walk/pard3e2/no_mistranslation_popsize1e4/"
    misdir = "../results/adaptive_walk/pard3e2/proteins500_popsize1e4/"
    nomisprops = get_misrates_over_time(nomisdir)
    misprops = get_misrates_over_time(misdir)
    # mistranslation rates of the landscape
    misrates = np.loadtxt("../data/pard3e2_mistranslation_rates.txt")
    proptime = get_percent_misrates_over_time(nomisprops, misprops, np.mean(misrates))

    # Panel B
    axB = plt.subplot2grid((1,2), (0,1), colspan=1)
    for mis, vals in proptime.items():
        x_pos, means, stderr = vals
        if mis.startswith("without"):
            c = wes(0.9999)
        else:
            c = wes(0.0)
        axB.fill_between(x_pos, means-1.96*stderr, means+1.96*stderr, alpha=0.25,
            color=c)
        axB.plot(x_pos, means, color=c, label=mis)
    axB.set_xlabel("Number of mutations", fontsize=16)
    axB.set_ylabel("", fontsize=16)
    axB.set_xticks(np.linspace(0, max(x_pos), 5))
    axB.tick_params(axis='both', which='major', labelsize=14)
    axB.set_xlim(0,max(x_pos))
    axB.set_ylim(98.0,100.5)
    axB.annotate("B", xy=(0,1), xycoords="axes fraction",
                xytext=(5,-5), textcoords="offset points",
                ha="left", va="top", weight="bold", fontsize=25)
    fig.savefig("../writing/figures/figure_SI_6.png")
    plt.tight_layout()
    plt.close()

def plot_figure_SI_7():
    nearly_neutral = pd.read_csv("../results/neutral_networks/nearly_neutral_size.csv")
    pr = 1
    kind = "both"
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


###################################################################
# Helper functions
###################################################################

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

def get_mutnum_to_endpoint(directory):
    flist = [f for f in os.listdir(directory) if "walk" in f]
    num = []
    for f in sorted(flist):
        # number of fixation events until endpoint: length of file
        num.append(len(np.loadtxt(directory+f, dtype=str))-1)
    return num

def get_fitness_change_time(directory):
    # For those runs where more than one mutation became fixed
    # check if the first fixation event is larger for mistr or no mistr
    # Also compare the fitness benefits of the last fixation event
    # Or try to plot the whole trajectory in between??
    flist = [f for f in os.listdir(directory) if "fitness" in f]
    chfit = []
    mutnum = []
    synon = []
    for f in sorted(flist):
        traj = np.loadtxt(directory+f)
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
            for row in range(len(data)-1):
                times.append(data[row+1][0])
                chf.append(data[row+1][2] - data[row][2])

                prev = data[row][1]
                mut = data[row+1][1]
                if evo.hamming(evo.translate(prev), evo.translate(mut))==0:
                    sy.append(True)
                else:
                    sy.append(False)
            # Get all runs toegether
            synon.append(sy)
            chfit.append(chf)
            mutnum.append(times)
    return chfit, mutnum, synon

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

def get_percent_misrates_over_time(nomisprops, misprops, meanmisr):
    proptime = {}
    for vals,mis in zip([nomisprops, misprops], ["without mistranslation", "with mistranslation"]):
        yraw = np.array([v[0]/meanmisr*100 for v in vals])
        xraw = np.array([v[1] for v in vals])
        x_pos = np.unique(xraw)
        means = np.array([np.mean(yraw[xraw==v]) for v in x_pos])
        stdv = np.array([np.std(yraw[xraw==v]) for v in x_pos])
        proptime[mis] = [x_pos, means, stdv/np.sqrt(means)]
    return proptime

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

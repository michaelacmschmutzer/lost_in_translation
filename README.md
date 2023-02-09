
This repository contains the model described in "Not quite lost in translation: Mistranslation alters adaptive landscape topography and the dynamics of evolution" (Schmutzer
& Wagner). All code is written for python3. The files and their contents are:

analysis.py                       Analysis and exploratory code
adaptive_walk.py                  Simulate adaptive walk through a landscape (to be run on a parallel computing cluster)
build_landscapes.py               Read in data and assemble dictionaries mapping protein sequence to fitness
cluster_fitness.py                Calculate the fitness and effective population size of genotypes (to be run on a parallel computing cluster)
cluster_mistranslation.py         Run cluster_fitness on cluster.
evolution.py                      The main workhorse. Contains functions for simulating mutation, fixation (or loss) of mutations, and for estimating the fitness of genotypes.
figure_plotting.py                Plots publication figures.
makedatabase.py                   Makes SQL database of genotype sequence vs predicted fitness in the presence of mistranslation (output of cluster_fitness.py)
mistranslation_rate_estimate.py.  Takes mistranslation rate measurements from Mordret et al. (2019). Interpolates rare mistranslation rates.
neutral_networks.py               Searches for neutral networks in fitness landscapes. Determines degree of epistasis.
selection_power.py                Calculate the power of selection between two genotypes in the presence and absence of mistranslation (cluster)
test_evolution.py                 Unit-tests. Test core functions of evolution.py

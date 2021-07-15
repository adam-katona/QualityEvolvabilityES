# Overview

This repository is code accompanying the paper:

[Quality Evolvability ES: Evolving Individuals With a Distribution of Well Performing and Diverse Offspring](https://arxiv.org/abs/2103.10790), Adam Katona, Daniel W. Franks and  James Alfred Walker


Quality Evolvability ES simultaniously optimizes fitness and evolvability, where evolvability is measures as the behavioural variance of offspring. Quality Evolvability forces the solution to become evolvable, unlike Quality Diversity, which aims to learn an archive of diverse, but potentially genetically distant individuals. Evolution of evolvability is desirable because it allows the possibility of accumulating the ability to evolve for a long time, potentially enabling the speed up of future evolution by orders of magnitude, allowing us to utilize evolution in areas that were not practical before.


Contents of this repository:
- simple_implementation: A simple implementation of Quality Evolvability ES
- paper_code: Original code used in the paper 

## Simple implementation of Quality Evolvability ES

A simple implementation of Quality Evolvability ES, to make it easy to understand the algorithm in its simplest from (without the complexities of running on a distributed cluster)

The simple implementation consist of the files:
- qees.py   (the algorithm with a simple ask() tell() interface)
- test_qees.py  (example of applying QE-ES to a simple problem)


## Code for experiments in paper

Our paper includes experiments with various modified versions of the pybullet Ant and Humanoid environments,
with algorithms ES, maxvar Evolvability ES, and our proposed Quality Evolvability ES.

The code was originally forked from https://github.com/uber-research/Evolvability-ES, which includes experiments with
the Cheetah and Ant environments with ES, maxvar Evolvability ES and maxent Evolvability ES.

We added:
- our new algorithm Quality Evolvability ES
- new modified versions of environments (Ant, Humanoid), (Normal, Directional, Deceptive) 
- code to run on SLURM using dask's slurm_jobqueue library

Some important files:
Main functions:
- main.py                Contains the main function to run a single experiment (various env, various model, various algo)
- run_with_jobfile.py    Creates a cluster and runs the main function from main.py
- run.py                 Creates a list of jobs, and keep submitting them (by calling run_with_jobfile.py) as free spots become available.

Algorithms
- distributed_evolution/gen_es.py               The ES implemenation from uber ai
- distributed_evolution/gen_var_es.py           The Evolvability ES implemenation from uber ai
- distributed_evolution/gen_nondominated_es.py  Our Quality Evolvability ES implementation

Environment
- distributed_evolution/envs/ant.py          Our env variations for Ant
- distributed_evolution/envs/humanoid.py     Our env variations for Humanoid

Policy
- distributed_evolution/policies/my_torch_policies.py   The policy implementation in pytorch





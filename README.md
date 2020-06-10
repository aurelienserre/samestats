# samestats

The goal here is to propose a new algorithm to transform datasets while keeping some of their statistical properties unchanged.
Basically, the algo takes a dataset as input, as well as a target shape, and outputs a new dataset that approxiamtely matches the target shape, while having the same statistical proerties as the original one.
For that, I try to use gradient descent, with a carefully crafted objective function.
This is work in progress.

## Background
It all started with Anscombe publishing in 1973 a paper containing what is now called [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet).
Anscombe's quartet is simply a set of four 2D datasets, each made of 11 points, that look very different, but share many statistical properties.
Part of Anscombe's message in his paper is that on top of looking at summary statistics, it is also very important to visualize the data when dealing with a new dataset and trying to make sense of it.

Then the authors of [this paper](https://www.autodeskresearch.com/publications/samestats) noted that Anscombe never explained how the four datasets were generated, and they propose an algorithm for "Generating Datasets with Varied Appearance and Identical Statistics".
The algorithm they propose is based on the idea of "Simulated Annealing".

When looking at the algorithm they propose, I thought it might be possible to derive a more efficient algorithm, since the objective function implicitly being optimized is rather smooth, despite the search space being possibly quite large.
I thought I would be fun to try an use gradient descent for this task (maybe not the most obvious optimization technique given that there are constraints to respect).

I am using `pytorch` for computing the gradient, and performing the gradient descent.

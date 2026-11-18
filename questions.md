I introduce you to the big picture:
We have developed an extension for this QCNC paper ~/Research/qcnc2025-paper
which is built on top of this TQE paper /Documents/SoA/efficient-optimization-of-cutoffs-in-quantum-repeater-chains
which was the base for this current work repository
We have extended the code according to new representation of bi-partite entangled states, as explained in ~/Research/lilla-overleaf

In /Users/lorenzolacorte/Research/lilla/src/analysis-i there are some tests we did to explore several insights that arise when evaluating protocols under non-unital noise.
My question is: some work like https://arxiv.org/abs/2603.02610 use a Tuning parameter \Beta to control the trade-off between rate and fidelity of entanglement generation. We could use something similar in our Bayesian optimization machinery. However, I would like a direct reference to this trade-off, specifically with using this Partially-Entangled sources with a q parameter.

Then the final objective would be **extending the results of the original papers on cutoffs, swapping order, distillation placement, and possibly combine all those** and along with one of the new thing “classical communication time” can be good for our work. This means extending the Bayesian optimization script with a cutoff parameter choise (we could argue a uniform memory cutoff as it shows best performance in /Documents/SoA/efficient-optimization-of-cutoffs-in-quantum-repeater-chains, right?)
Even without the comparison of “how much we lose by twirling”. That can be a separate analysis in some later work if we can think of.
And finally the post-selection thing and the partially entangled states can also be good addition I think: it can be add to the optimization machinery with a tuning parameter.
Sampling from Population
=========================

The *Distributions* package provides functions for sampling from a given population in a variety ways.

Sampling with/without Replacement
----------------------------------

.. function:: sample(a)

	Randomly draw an element from the source array a.

.. function:: sample(a, n[, replace=true])

	Randomly draw n samples from the source array a with replacement (by default, ``replace`` is set to be true). This function returns an array with n elements. 

	To sample without replacement, one can set the keyword argument ``replace`` to be false.

.. function:: sample(a, dims[, replace=true])

	This function returns an array of samples, whose size is specified by dims.

.. function:: sample!(a, x[, replace=true])

	Draw n samples from a and write them to x. The keyword argument ``replace`` controls whether the sampling is with or without replacement.


Weighted Sampling
-------------------

.. function:: wsample(a, w)

	Draw a sample from a, such that the chance of ``a[i]`` being drawn is proportional to its weight (given by ``w[i]``).

.. function:: wsample(a, w, wsum)

	Draw a sample from a, with probabilities proportional to weights in w. This method allows the user to supply the sum of weights (if it is known or has been computed in advance). This saves the time of computing the sum of weights again within ``wsample``. 

.. function:: wsample(a, w, n[, wsum=NaN])

	Draw n samples from a, with probabilities proportional to the weights in w. The keyword argument ``wsum`` allows the user to supply a pre-computed sum of weights. When ``wsum`` is NaN, this function will compute the sum of weights internally.

.. function:: wsample(a, w, dims[, wsum=NaN])

	Draw samples from a, with probabilities proportional to the weights in w. This function returns an array whose size is specified by dims.

.. function:: wsample!(a, w, x[, wsum=NaN])

	Draw samples from a, with probabilities proportional to the weights in w, and write them to a pre-allocated array x. 


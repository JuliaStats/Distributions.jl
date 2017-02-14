.. _univariates:

Univariate Distributions
==========================


..  toctree::
    :maxdepth: 1

    univariate-continuous.rst
    univariate-discrete.rst
    
*Univariate distributions* are the distributions whose variate forms are ``Univariate`` (*i.e* each sample is a scalar). Abstract types for univariate distributions:

.. code-block:: julia

    const UnivariateDistribution{S<:ValueSupport} = Distribution{Univariate,S}

    const DiscreteUnivariateDistribution   = Distribution{Univariate, Discrete}
    const ContinuousUnivariateDistribution = Distribution{Univariate, Continuous}


Common Interface
------------------

A series of methods are implemented for each univariate distribution, which provide useful functionalities such as moment computation, pdf evaluation, and sampling (*i.e.* random number generation).

Parameter Retrieval
~~~~~~~~~~~~~~~~~~~~~~

.. function:: params(d)

    Return a tuple of parameters.

    **Note:** Let ``d`` be a distribution of type ``D``, then ``D(params(d)...)`` will construct exactly the same distribution as ``d``.

.. function:: succprob(d)

    Get the probability of success.

.. function:: failprob(d)

    Get the probability of failure.

.. function:: scale(d)

    Get the scale parameter.

.. function:: location(d)

    Get the location parameter.

.. function:: shape(d)

    Get the shape parameter.

.. function:: rate(d)

    Get the rate parameter.

.. function:: ncategories(d)

    Get the number of categories.

.. function:: ntrials(d)

    Get the number of trials.

.. function:: dof(d)

    Get the degrees of freedom.


**Note:** ``params`` are defined for all univariate distributions, while other parameter retrieval methods are only defined for those distributions for which these parameters make sense. See below for details.


Computation of statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let ``d`` be a distribution:

.. function:: mean(d)

    Return the expectation of distribution ``d``.

.. function:: var(d)

    Return the variance of distribution ``d``.

.. function:: std(d)

    Return the standard deviation of distribution ``d``, i.e. ``sqrt(var(d))``.

.. function:: median(d)

    Return the median value of distribution ``d``.

.. function:: modes(d)

    Return an array of all modes of ``d``.

.. function:: mode(d)

    Return the mode of distribution ``d``. If ``d`` has multiple modes, it returns the first one, i.e. ``modes(d)[1]``.

.. function:: skewness(d)

    Return the skewness of distribution ``d``.

.. function:: kurtosis(d)

    Return the excess kurtosis of distribution ``d``.

.. function:: isplatykurtic(d)

    Return whether ``d`` is platykurtic (*i.e* ``kurtosis(d) > 0``).

.. function:: isleptokurtic(d)

    Return whether ``d`` is leptokurtic (*i.e* ``kurtosis(d) < 0``).

.. function:: ismesokurtic(d)

    Return whether ``d`` is mesokurtic (*i.e* ``kurtosis(d) == 0``).

.. function:: entropy(d)

    Return the entropy value of distribution ``d``.

.. function:: entropy(d, base)

    Return the entropy value of distribution ``d``, w.r.t. a given base.

.. function:: mgf(d, t)

    Evaluate the moment generating function of distribution ``d``.

.. function:: cf(d, t)

    Evaluate the characteristic function of distribution ``d``.

Probability Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: insupport(d, x)

    When ``x`` is a scalar, it returns whether x is within the support of ``d``.
    When ``x`` is an array, it returns whether every element in x is within the support of ``d``.

.. function:: pdf(d, x)

    The pdf value(s) evaluated at ``x``.

.. function:: pdf(d, rgn)

    Get/compute the probabilities over a range of values. Here, ``rgn`` should be in the form of ``a:b``.

    **Note:** computing the probabilities over a contiguous range of values can take advantage of the recursive relations between probability masses and thus is often more efficient than computing these probabilities individually.

.. function:: pdf(d)

    Get/compute the entire probability vector of ``d``. This is equivalent to ``pdf(d, minimum(d):maximum(d))``.

    **Note:** this method is only defined for *bounded* distributions.


.. function:: logpdf(d, x)

    The logarithm of the pdf value(s) evaluated at x, i.e. ``log(pdf(x))``.

    **Note:** The internal implementation may directly evaluate logpdf instead of first computing pdf and then taking the logarithm, for better numerical stability or efficiency.

.. function:: loglikelihood(d, x)

    The log-likelihood of distribution ``d`` w.r.t. all samples contained in array ``x``.

.. function:: cdf(d, x)

    The cumulative distribution function evaluated at ``x``.

.. function:: logcdf(d, x)

    The logarithm of the cumulative function value(s) evaluated at ``x``, i.e. ``log(cdf(x))``.

.. function:: ccdf(d, x)

    The complementary cumulative function evaluated at ``x``, i.e. ``1 - cdf(d, x)``.

.. function:: logccdf(d, x)

    The logarithm of the complementary cumulative function values evaluated at x, i.e. ``log(ccdf(x))``.

.. function:: quantile(d, q)

    The quantile value. Let ``x = quantile(d, q)``, then ``cdf(d, x) = q``.

.. function:: cquantile(d, q)

    The complementary quantile value, i.e. ``quantile(d, 1-q)``.

.. function:: invlogcdf(d, lp)

    The inverse function of logcdf.

.. function:: invlogccdf(d, lp)

    The inverse function of logccdf.


Vectorized evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Vectorized computation and inplace vectorized computation are supported for the following functions:

* ``pdf``
* ``logpdf``
* ``cdf``
* ``logcdf``
* ``ccdf``
* ``logccdf``
* ``quantile``
* ``cquantile``
* ``invlogcdf``
* ``invlogccdf``

For example, when ``x`` is an array, then ``r = pdf(d, x)`` returns an array ``r`` of the same size, such that ``r[i] = pdf(d, x[i])``. One can also use ``pdf!`` to write results to pre-allocated storage, as ``pdf!(r, d, x)``.


Sampling (Random number generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: rand(d)

    Draw a sample from d

.. function:: rand(d, n)

    Return a vector comprised of n independent samples from the distribution ``d``.

.. function:: rand(d, dims)

    Return an array of size ``dims`` that is filled with independent samples from the distribution ``d``.

.. function:: rand(d, dim0, dim1, ...)

    Similar to ``rand(d, dims)`` above, except that the dimensions can be input as individual integers.

    For example, one can write ``rand(d, 2, 3)`` or ``rand(d, (2, 3))``, which are equivalent.

.. function:: rand!(d, arr)

    Fills a pre-allocated array ``arr`` with independent samples from the distribution ``d``.



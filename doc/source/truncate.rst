.. _truncate:

Truncated Distributions
========================

The package provides a type, named `Truncated`, to represented truncated distributions, which is defined as below:

.. code:: julia

    immutable Truncated{D<:UnivariateDistribution,S<:ValueSupport} <: Distribution{Univariate,S}
        untruncated::D      # the original distribution (untruncated)
        lower::Float64      # lower bound
        upper::Float64      # upper bound
        lcdf::Float64       # cdf of lower bound
        ucdf::Float64       # cdf of upper bound

        tp::Float64         # the probability of the truncated part, i.e. ucdf - lcdf
        logtp::Float64      # log(tp), i.e. log(ucdf - lcdf)
    end


A truncated distribution can be constructed using the constructor ``Truncated`` as follows:

.. function:: Truncated(d, l, u):

    Construct a truncated distribution.

    :param d:   The original distribution.
    :param l:   The lower bound of the truncation, which can be a finite value or `-Inf`.
    :param u:   The upper bound of the truncation, which can be a finite value of `Inf`.


Many functions, including those for the evaluation of pdf and sampling, are defined for all truncated univariate distributions:

    - ``maximum``
    - ``minimum``
    - ``insupport``
    - ``pdf``
    - ``logpdf``
    - ``cdf``
    - ``logcdf``
    - ``ccdf``
    - ``logccdf``
    - ``quantile``
    - ``cquantile``
    - ``invlogcdf``
    - ``invlogccdf``
    - ``rand``
    - ``rand!``
    - ``median``

However, functions to compute statistics, such as ``mean``, ``mode``, ``var``, ``std``, and ``entropy``, are not available for generic truncated distributions. Generally, there are no easy ways to compute such quantities due to the complications incurred by truncation.


Truncated Normal Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *truncated normal distribution* is a particularly important one in the family of truncated distributions. We provide additional support for this type.

One can construct a truncated normal distribution using the common constructor `Truncated`, as

.. code:: julia

    Truncated(Normal(mu, sigma), l, u)

or using a dedicated constructor `TruncatedNormal` as

.. code:: julia

    TruncatedNormal(mu, sigma, l, u)

Also, we provide additional methods to compute various statistics for truncated normal:

    - ``mean``
    - ``mode``
    - ``modes``
    - ``var``
    - ``std``
    - ``entropy``


.. _locationscale:

Location Scale families
========================

The package provides a type named `LocationScale`, to represent distributions from a location-scale family, which is defined as:

.. code:: julia
    immutable LocationScale{D<:UnivariateDistribution, S<:ValueSupport} <: UnivariateDistribution{S}
      base::D
      μ::Float64
      σ::Float64
    end


A location-scale distribution can be constructed using the constructor ``LocationScale`` as follows:

.. function:: LocationScale(d, μ, σ):

    Construct a location scale distribution.

    :param d:   The base distribution for the family
    :param μ:   The location parameter
    :param σ:   The scale parameters


Additionally, location-scale distributions can be constructed using arithmetic operations which are defined as:

.. code:: julia
    +(d::LocationScale, μ::Real) = LocationScale(d.base, d.μ + μ, d.σ)
    -(d::LocationScale, μ::Real) = LocationScale(d.base, d.μ - μ, d.σ)
    *(d::LocationScale, σ::Real) = LocationScale(d.base, d.μ*σ, d.σ*σ)
    /(d::LocationScale, σ::Real) = LocationScale(d.base, d.μ/σ, d.σ/σ)


    +(μ::Real, d::LocationScale) = d + μ
    *(μ::Real, d::LocationScale) = d * μ

    +(d::UnivariateDistribution, μ::Real) = LocationScale(d, μ, 1.0)
    -(d::UnivariateDistribution, μ::Real) = LocationScale(d, -μ, 1.0)
    *(d::UnivariateDistribution, σ::Real) = LocationScale(d, 0.0, σ)
    /(d::UnivariateDistribution, σ::Real) = LocationScale(d, 0.0, 1/σ)


    +(μ::Real, d::UnivariateDistribution) = d + μ
    *(μ::Real, d::UnivariateDistribution) = d * μ



Many functions, including those for the evaluation of pdf and sampling, are defined for all location-scale univariate distribuitons:

    - ``maximum``
    - ``minimum``
    - ``insupport``
    - ``location``
    - ``scale``
    - ``pdf``
    - ``logpdf``
    - ``cdf``
    - ``quantile``
    - ``rand``
    - ``mean``
    - ``var``
    - ``skewness``
    - ``kurtosis``
    - ``rand``


Observe that some distributions are natively parameterized as location--scale families, namely:

    - ``Cauchy``
    - ``GeneralizedExtremeValue``
    - ``GeneralizedPareto``
    - ``Gumbel``
    - ``Laplace``
    - ``Levy``
    - ``Normal`
    - ``SymTriangularDist``
    - ``Weibull``


Distributions can be standardized with the function ``standardize`` as follows:

.. function:: standardize(d):

    Center and scale the distribution ``d`` to have zero mean and unit variance.

    :param d:   The distribution we wish to standardize

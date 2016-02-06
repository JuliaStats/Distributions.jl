.. _multivariates:

Multivariate Distributions
=============================

*Multivariate distributions* are the distributions whose variate forms are ``Multivariate`` (*i.e* each sample is a vector). Abstract types for multivariate distributions:

.. code-block:: julia

    typealias MultivariateDistribution{S<:ValueSupport} Distribution{Multivariate,S}

    typealias DiscreteMultivariateDistribution   Distribution{Multivariate, Discrete}
    typealias ContinuousMultivariateDistribution Distribution{Multivariate, Continuous}


Common Interface
------------------

The methods listed as below are implemented for each multivariate distribution, which provides a consistent interface to work with multivariate distributions.

Computation of statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: length(d)

    Return the sample dimension of distribution ``d``.

.. function:: size(d)

    Return the sample size of distribution ``d``, *i.e* ``(length(d), 1)``.

.. function:: mean(d)

    Return the mean vector of distribution ``d``.

.. function:: var(d)

    Return the vector of component-wise variances of distribution ``d``.

.. function:: cov(d)

    Return the covariance matrix of distribution ``d``.

.. function:: cor(d)

    Return the correlation matrix of distribution ``d``.

.. function:: entropy(d)

    Return the entropy of distribution ``d``.


Probability evaluation
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: insupport(d, x)

    If ``x`` is a vector, it returns whether x is within the support of ``d``.
    If ``x`` is a matrix, it returns whether every column in ``x`` is within the support of ``d``.

.. function:: pdf(d, x)

    Return the probability density of distribution ``d`` evaluated at ``x``.

    - If ``x`` is a vector, it returns the result as a scalar.
    - If ``x`` is a matrix with n columns, it returns a vector ``r`` of length n, where ``r[i]`` corresponds to ``x[:,i]`` (i.e. treating each column as a sample).

.. function:: pdf!(r, d, x)

    Evaluate the probability densities at columns of x, and write the results to a pre-allocated array r.

.. function:: logpdf(d, x)

    Return the logarithm of probability density evaluated at ``x``.

    - If ``x`` is a vector, it returns the result as a scalar.
    - If ``x`` is a matrix with n columns, it returns a vector ``r`` of length n, where ``r[i]`` corresponds to ``x[:,i]``.

.. function:: logpdf!(r, d, x)

    Evaluate the logarithm of probability densities at columns of x, and write the results to a pre-allocated array r.

.. function:: loglikelihood(d, x)

    The log-likelihood of distribution ``d`` w.r.t. all columns contained in matrix ``x``.

**Note:** For multivariate distributions, the pdf value is usually very small or large, and therefore direct evaluating the pdf may cause numerical problems. It is generally advisable to perform probability computation in log-scale.


Sampling
~~~~~~~~~

.. function:: rand(d)

    Sample a vector from the distribution ``d``.

.. function:: rand(d, n)

    Sample n vectors from the distribution ``d``. This returns a matrix of size ``(dim(d), n)``, where each column is a sample.

.. function:: rand!(d, x)

    Draw samples and output them to a pre-allocated array x. Here, x can be either a vector of length ``dim(d)`` or a matrix with ``dim(d)`` rows.


**Node:** In addition to these common methods, each multivariate distribution has its own special methods, as introduced below.


.. _multinomial:

Multinomial Distribution
---------------------------

The `Multinomial distribution <http://en.wikipedia.org/wiki/Multinomial_distribution>`_ generalizes the *binomial distribution*. Consider n independent draws from a Categorical distribution over a finite set of size k, and let :math:`X = (X_1, ..., X_k)` where ``X_i`` represents the number of times the element ``i`` occurs, then the distribution of ``X`` is a multinomial distribution. Each sample of a multinomial distribution is a k-dimensional integer vector that sums to n.

The probability mass function is given by

.. math::

    f(x; n, p) = \frac{n!}{x_1! \cdots x_k!} \prod_{i=1}^k p_i^{x_i},
    \quad x_1 + \cdots + x_k = n

.. code-block:: julia

    Multinomial(n, p)   # Multinomial distribution for n trials with probability vector p

    Multinomial(n, k)   # Multinomial distribution for n trials with equal probabilities
                        # over 1:k


.. _multivariatenormal:

Multivariate Normal Distribution
----------------------------------

The `Multivariate normal distribution <http://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_ is a multidimensional generalization of the *normal distribution*. The probability density function of a d-dimensional multivariate normal distribution with mean vector :math:`\boldsymbol{\mu}` and covariance matrix :math:`\boldsymbol{\Sigma}` is

.. math::

    f(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2 \pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}}
    \exp \left( - \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)

We realize that the mean vector and the covariance often have special forms in practice, which can be exploited to simplify the computation. For example, the mean vector is sometimes just a zero vector, while the covariance matrix can be a diagonal matrix or even in the form of :math:`\sigma \mathbf{I}`. To take advantage of such special cases, we introduce a parametric type ``MvNormal``, defined as below, which allows users to specify the special structure of the mean and covariance.

.. code-block:: julia

    immutable MvNormal{Cov<:AbstractPDMat,Mean<:Union{Vector{Float64},ZeroVector{Float64}}} <: AbstractMvNormal
        μ::Mean
        Σ::Cov
    end

Here, the mean vector can be an instance of either ``Vector{Float64}`` or ``ZeroVector{Float64}``, where the latter is simply an empty type indicating a vector filled with zeros. The covariance can be of any subtype of ``AbstractPDMat``. Particularly, one can use ``PDMat`` for full covariance, ``PDiagMat`` for diagonal covariance, and ``ScalMat`` for the isotropic covariance -- those in the form of :math:`\sigma \mathbf{I}`. (See the Julia package `PDMats <https://github.com/lindahua/PDMats.jl>`_ for details).

We also define a set of alias for the types using different combinations of mean vectors and covariance:

.. code-block:: julia

    typealias IsoNormal  MvNormal{ScalMat,  Vector{Float64}}
    typealias DiagNormal MvNormal{PDiagMat, Vector{Float64}}
    typealias FullNormal MvNormal{PDMat,    Vector{Float64}}

    typealias ZeroMeanIsoNormal  MvNormal{ScalMat,  ZeroVector{Float64}}
    typealias ZeroMeanDiagNormal MvNormal{PDiagMat, ZeroVector{Float64}}
    typealias ZeroMeanFullNormal MvNormal{PDMat,    ZeroVector{Float64}}


Construction
~~~~~~~~~~~~~

Generally, users don't have to worry about these internal details. We provide a common constructor ``MvNormal``, which will construct a distribution of appropriate type depending on the input arguments.

.. function:: MvNormal(mu, sig)

    Construct a multivariate normal distribution with mean ``mu`` and covariance represented by ``sig``.

    :param mu:      The mean vector, of type ``Vector{Float64}``.
    :param sig:     The covariance, which can in of either of the following forms:

                    - an instance of a subtype of ``AbstractPDMat``
                    - a symmetric matrix of type ``Matrix{Float64}``
                    - a vector of type ``Vector{Float64}``: indicating a diagonal covariance as ``diagm(abs2(sig))``.
                    - a real-valued number: indicating an isotropic covariance as ``abs2(sig) * eye(d)``.

.. function:: MvNormal(sig)

    Construct a multivariate normal distribution with zero mean and covariance represented by ``sig``.

    Here, ``sig`` can be in either of the following forms:

    - an instance of a subtype of ``AbstractPDMat``
    - a symmetric matrix of type ``Matrix{Float64}``
    - a vector of type ``Vector{Float64}``: indicating a diagonal covariance as ``diagm(abs2(sig))``.


.. function:: MvNormal(d, sig)

    Construct a multivariate normal distribution of dimension ``d``, with zero mean, and an isotropic covariance as ``abs2(sig) * eye(d)``.


**Note:** The constructor will choose an appropriate covariance form internally, so that special structure of the covariance can be exploited.


Addition Methods
~~~~~~~~~~~~~~~~~

In addition to the methods listed in the common interface above, we also provide the followinig methods for all multivariate distributions under the base type ``AbstractMvNormal``:

.. function:: invcov(d)

    Return the inversed covariance matrix of d.

.. function:: logdetcov(d)

    Return the log-determinant value of the covariance matrix.

.. function:: sqmahal(d, x)

    Return the squared Mahalanobis distance from x to the center of d, w.r.t. the covariance.

    When x is a vector, it returns a scalar value. When x is a matrix, it returns a vector of length size(x,2).

.. function:: sqmahal!(r, d, x)

    Write the squared Mahalanbobis distances from each column of x to the center of d to r.


Canonical form
~~~~~~~~~~~~~~~

Multivariate normal distribution is an `exponential family distribution <http://en.wikipedia.org/wiki/Exponential_family>`_, with two *canonical parameters*: the *potential vector* :math:`\mathbf{h}` and the *precision matrix* :math:`\mathbf{J}`. The relation between these parameters and the conventional representation (*i.e.* the one using mean :math:`\boldsymbol{mu}` and covariance :math:`\boldsymbol{\Sigma}`) is:

.. math::

    \mathbf{h} = \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}, \quad \text{ and } \quad \mathbf{J} = \boldsymbol{\Sigma}^{-1}

The canonical parameterization is widely used in Bayesian analysis. We provide a type ``MvNormalCanon``, which is also a subtype of ``AbstractMvNormal`` to represent a multivariate normal distribution using canonical parameters. Particularly, ``MvNormalCanon`` is defined as:

.. code:: julia

    immutable MvNormalCanon{P<:AbstractPDMat,V<:Union{Vector{Float64},ZeroVector{Float64}}} <: AbstractMvNormal
        μ::V    # the mean vector
        h::V    # potential vector, i.e. inv(Σ) * μ
        J::P    # precision matrix, i.e. inv(Σ)
    end

We also define aliases for common specializations of this parametric type:

.. code:: julia

    typealias FullNormalCanon MvNormalCanon{PDMat,    Vector{Float64}}
    typealias DiagNormalCanon MvNormalCanon{PDiagMat, Vector{Float64}}
    typealias IsoNormalCanon  MvNormalCanon{ScalMat,  Vector{Float64}}

    typealias ZeroMeanFullNormalCanon MvNormalCanon{PDMat,    ZeroVector{Float64}}
    typealias ZeroMeanDiagNormalCanon MvNormalCanon{PDiagMat, ZeroVector{Float64}}
    typealias ZeroMeanIsoNormalCanon  MvNormalCanon{ScalMat,  ZeroVector{Float64}}

A multivariate distribution with canonical parameterization can be constructed using a common constructor ``MvNormalCanon`` as:

.. function:: MvNormalCanon(h, J)

    Construct a multivariate normal distribution with potential vector ``h`` and precision matrix represented by ``J``.

    :param h:   the potential vector, of type ``Vector{Float64}``.
    :param J:   the representation of the precision matrix, which can be in either of the following forms:

                - an instance of a subtype of ``AbstractPDMat``
                - a square matrix of type ``Matrix{Float64}``
                - a vector of type ``Vector{Float64}``: indicating a diagonal precision matrix as ``diagm(J)``.
                - a real number: indicating an isotropic precision matrix as ``J * eye(d)``.

.. function:: MvNormalCanon(J)

    Construct a multivariate normal distribution with zero mean (thus zero potential vector) and precision matrix represented by ``J``.

    Here, ``J`` represents the precision matrix, which can be in either of the following forms:

    - an instance of a subtype of ``AbstractPDMat``
    - a square matrix of type ``Matrix{Float64}``
    - a vector of type ``Vector{Float64}``: indicating a diagonal precision matrix as ``diagm(J)``.


.. function:: MvNormalCanon(d, v)

    Construct a multivariate normal distribution of dimension ``d``, with zero mean and a precision matrix as ``v * eye(d)``.

**Note:** ``MvNormalCanon`` share the same set of methods as ``MvNormal``.

.. _multivariatelognormal:

Multivariate Lognormal Distribution
-----------------------------------

The `Multivariate lognormal distribution <http://en.wikipedia.org/wiki/Log-normal_distribution>`_ is a multidimensional generalization of the *lognormal distribution*.

If :math:`\boldsymbol X \sim \mathcal{N}(\boldsymbol\mu,\,\boldsymbol\Sigma)` has a multivariate normal distribution then :math:`\boldsymbol Y=\exp(\boldsymbol X)` has a multivariate lognormal distribution.

Mean vector :math:`\boldsymbol{\mu}` and covariance matrix :math:`\boldsymbol{\Sigma}` of the underlying normal distribution are known as the *location* and *scale* parameters of the corresponding lognormal distribution.

The package provides an implementation, ``MvLogNormal``, which wraps around ``MvNormal``:

.. code-block:: julia

    immutable MvLogNormal <: AbstractMvLogNormal
      normal::MvNormal
    end

Construction
~~~~~~~~~~~~

``MvLogNormal`` provides the same constructors as ``MvNormal``. See above for details.

Additional Methods
~~~~~~~~~~~~~~~~~~

In addition to the methods listed in the common interface above, we also provide the following methods:

.. function:: location(d)

    Return the location vector of the distribution (the mean of the underlying normal distribution).

.. function:: scale(d)

    Return the scale matrix of the distribution (the covariance matrix of the underlying normal distribution).

.. function:: median(d)

    Return the median vector of the lognormal distribution. which is strictly smaller than the mean.

.. function:: mode(d)

    Return the mode vector of the lognormal distribution, which is strictly smaller than the mean and median.

Conversion Methods
~~~~~~~~~~~~~~~~~~

It can be necessary to calculate the parameters of the lognormal (location vector and scale matrix) from a given covariance and mean, median or mode. To that end, the following functions are provided.

.. function:: location{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix)

    Calculate the location vector (the mean of the underlying normal distribution).

    If ``s == :meancov``, then m is taken as the mean, and S the covariance matrix of a lognormal distribution.

    If ``s == :mean | :median | :mode``, then m is taken as the mean, median or mode of the lognormal respectively, and S is interpreted as the scale matrix (the covariance of the underlying normal distribution).

    It is not possible to analytically calculate the location vector from e.g., median + covariance, or from mode + covariance.

.. function:: location!{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,μ::AbstractVector)

    Calculate the location vector (as above) and store the result in ``μ``

.. function:: scale{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix)

    Calculate the scale parameter, as defined for the location parameter above.

.. function:: scale!{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix)

    Calculate the scale parameter, as defined for the location parameter above and store the result in ``Σ``.

.. function:: params{D<:AbstractMvLogNormal}(::Type{D},m::AbstractVector,S::AbstractMatrix)

    Return (scale,location) for a given mean and covariance

.. function:: params!{D<:AbstractMvLogNormal}(::Type{D},m::AbstractVector,S::AbstractMatrix,μ::AbstractVector,Σ::AbstractMatrix)

    Calculate (scale,location) for a given mean and covariance, and store the results in ``μ`` and ``Σ``


.. _dirichlet:

Dirichlet Distribution
------------------------

The `Dirichlet distribution <http://en.wikipedia.org/wiki/Dirichlet_distribution>`_ is often used the conjugate prior for Categorical or Multinomial distributions. The probability density function of a Dirichlet distribution with parameter :math:`\alpha = (\alpha_1, \ldots, \alpha_k)` is

.. math::

    f(x; \alpha) = \frac{1}{B(\alpha)} \prod_{i=1}^k x_i^{\alpha_i - 1}, \quad \text{ with }
    B(\alpha) = \frac{\prod_{i=1}^k \Gamma(\alpha_i)}{\Gamma \left( \sum_{i=1}^k \alpha_i \right)},
    \quad x_1 + \cdots + x_k = 1


.. code-block:: julia

    # Let alpha be a vector
    Dirichlet(alpha)         # Dirichlet distribution with parameter vector alpha

    # Let a be a positive scalar
    Dirichlet(k, a)          # Dirichlet distribution with parameter a * ones(k)







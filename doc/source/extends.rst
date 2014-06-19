Create New Samplers and Distributions
=======================================

Whereas this package already provides a large collection of common distributions out of box, there are still occasions where you want to create new distributions (*e.g* your application requires a special kind of distributions, or you want to contribute to this package).

Generally, you don't have to implement every API method listed in the documentation. This package provides a series of generic functions that turn a small number of internal methods into user-end API methods. What you need to do is to implement this small set of internal methods for your distributions.

**Note:** the methods need to be implemented are different for distributions of different variate forms.


Create a Sampler
------------------

Unlike a full fledged distributions, a sampler, in general, only provides limited functionalities, mainly to support sampling.

Univariate Sampler
~~~~~~~~~~~~~~~~~~~~~

To implement a univariate sampler, one can define a sub type (say ``Spl``) of ``Sampleable{Univariate,S}`` (where ``S`` can be ``Disrete`` or ``Continuous``), and provide a ``rand`` method, as

.. code-block:: julia

    function rand(s::Spl)
        # ... generate a single sample from s
    end

The package already implements a vectorized version of ``rand!`` and ``rand`` that repeatedly calls the he scalar version to generate multiple samples.

Multivariate Sampler
~~~~~~~~~~~~~~~~~~~~~~

To implement a multivariate sampler, one can define a sub type of ``Sampleable{Multivariate,S}``, and provide both ``length`` and ``_rand!`` methods, as

.. code-block:: julia

    Base.length(s::Spl) = ... # return the length of each sample

    function _rand!{T<:Real}(s::Spl, x::AbstractVector{T})
        # ... generate a single vector sample to x
    end

This function can assume that the dimension of ``x`` is correct, and doesn't need to perform dimension checking.

The package implements both ``rand`` and ``rand!`` as follows (which you don't need to implement in general):

.. code-block:: julia

    function _rand!(s::Sampleable{Multivariate}, A::DenseMatrix)
        for i = 1:size(A,2)
            _rand!(s, view(A,:,i))
        end
        return A
    end

    function rand!(s::Sampleable{Multivariate}, A::AbstractVector)
        length(A) == length(s) ||
            throw(DimensionMismatch("Output size inconsistent with sample length."))
        _rand!(s, A)
    end

    function rand!(s::Sampleable{Multivariate}, A::DenseMatrix)
        size(A,1) == length(s) || 
            throw(DimensionMismatch("Output size inconsistent with sample length."))
        _rand!(s, A)
    end

    rand{S<:ValueSupport}(s::Sampleable{Multivariate,S}) = 
        _rand!(s, Array(eltype(S), length(s)))

    rand{S<:ValueSupport}(s::Sampleable{Multivariate,S}, n::Int) = 
        _rand!(s, Array(eltype(S), length(s), n))

If there is a more efficient method to generate multiple vector samples in batch, one should provide the following method 

.. code-block:: julia

    function _rand!{T<:Real}(s::Spl, A::DenseMatrix{T})
        ... generate multiple vector samples in batch
    end

Remember that each *column* of A is a sample. 

Matrix-variate Sampler
~~~~~~~~~~~~~~~~~~~~~~~

To implement a multivariate sampler, one can define a sub type of ``Sampleable{Multivariate,S}``, and provide both ``size`` and ``_rand!`` method, as

.. code-block:: julia

    Base.size(s::Spl) = ... # the size of each matrix sample

    function _rand!{T<:Real}(s::Spl, x::DenseMatrix{T})
        # ... generate a single matrix sample to x
    end

Note that you can assume ``x`` has correct dimensions in ``_rand!`` and don't have to perform dimension checking, the generic ``rand`` and ``rand!`` will do dimension checking and array allocation for you.


Create a Univariate Distribution
---------------------------------

A univariate distribution type should be defined as a subtype of ``DiscreteUnivarateDistribution`` or ``ContinuousUnivariateDistribution``. 

Following methods need to be implemented for each univariate distribution type (say ``D``):

.. function:: rand(d::D)

    Generate a scalar sample from ``d``.

.. function:: sampler(d::D)

    It is often the case that a sampler relies on some quantities that may be pre-computed in advance (that are not parameters themselves). 

    If such a more efficient sampler exists, one should provide this ``sampler`` method, which would be used for batch sampling. 

    The general fallback is ``sampler(d::Distribution) = d``. 


.. function:: pdf(d::D, x::Real)

    Evaluate the probability density (mass) at ``x``. 

    Note: The package implements the following generic methods to evaluate pdf values in batch.

    - ``pdf!(dst::AbstractArray, d::D, x::AbstractArray)``
    - ``pdf(d::D, x::AbstractArray)``

    If there exists more efficient routine to evaluate pdf in batch (faster than repeatedly calling the scalar version of ``pdf``), then one can also provide a specialized method of ``pdf!``. The vectorized version of ``pdf`` simply delegats to ``pdf!``.

.. function:: logpdf(d::D, x::Real)

    Evaluate the logarithm of probability density (mass) at ``x``.

    Whereas there is a fallback implemented ``logpdf(d, x) = log(pdf(d, x))``. Relying on this fallback is not recommended in general, as it is prone to overflow or underflow. 

    Again, the package provides vectorized version of ``logpdf!`` and ``logpdf``. One may override ``logpdf!`` to provide more efficient vectorized evaluation.

.. function:: cdf(d::D, x::Real)

    Evaluate the cumulative probability at ``x``.

    The package provides generic functions to compute ``ccdf``, ``logcdf``, and ``logccdf`` in both scalar and vectorized forms. One may override these generic fallbacks if the specialized versions provide better numeric stability or higher efficiency.

.. function:: quantile(d::D, q::Real)

    Evaluate the inverse cumulative distribution function at ``q``. 

    The package provides generic functions to compute ``cquantile``, ``invlogcdf``, and ``invlogccdf`` in both scalar and vectorized forms. One may override these generic fallbacks if the specialized versions provide better numeric stability or higher efficiency.

    Also a generic ``median`` is provided, as ``median(d) = quantile(d, 0.5)``. However, one should implement a specialized version of ``median`` if it can be computed faster than ``quantile``.

.. function:: minimum(d::D)

    Return the minimum of the support of ``d``.

.. function:: maximum(d::D)

    Return the maximum of the support of ``d``.

.. function:: insupport(d::D, x::Real)

    Return whether ``x`` is within the support of ``d``. 

    Note a generic fallback as ``insupport(d, x) = minimum(d) <= x <= maximum(d)`` is provided. However, it is often the case that ``insupport`` can be done more efficiently, and a specialized ``insupport`` is thus desirable. You should also override this function if the support is composed of multiple disjoint intervals.

    Vectorized versions of ``insupport!`` and ``insupport`` are provided as generic fallbacks.

It is also recommended that one also implements the following statistics functions: 

- ``mean``: compute the expectation.
- ``var``:  compute the variance. (A generic ``std`` is provided as ``std(d) = sqrt(var(d))``).
- ``modes``: get all modes (if this makes sense).
- ``mode``: returns the first mode.
- ``skewness``: compute the skewness.
- ``kurtosis``: compute the excessive kurtosis.
- ``entropy``: compute the entropy.
- ``mgf``: compute the moment generating functions.
- ``cf``: compute the characteristic function.

You may refer to the source file ``src/univariates.jl`` to see details about how generic fallback functions for univariates are implemented. 


Create a Multivariate Distribution
-----------------------------------

A multivariate distribution type should be defined as a subtype of ``DiscreteMultivarateDistribution`` or ``ContinuousMultivariateDistribution``. 

Following methods need to be implemented for each univariate distribution type (say ``D``):

.. function:: length(d::D)

    Return the length of each sample (*i.e* the dimension of the sample space).

.. function:: _rand!{T<:Real}(d::D, x::AbstractVector{T})

    Generate a vector sample to ``x``. 

    This function does not need to perform dimension checking. 

.. function:: sampler(d::D)

    Return a sampler for efficient batch/repeated sampling.

.. function:: _logpdf{T<:Real}(d::D, x::AbstractVector{T})

    Evaluate logarithm of pdf value for a given vector ``x``. This function need not perform dimension checking.

    Generally, one does not need to implement ``pdf`` (or ``_pdf``). The package provides fallback methods as follows:

    .. code-block:: julia

        _pdf(d::MultivariateDistribution, x::AbstractVector) = exp(_logpdf(d, x))

        function logpdf(d::MultivariateDistribution, x::AbstractVector)
            length(d) == length(x) ||
                throw(DimensionMismatch("Inconsistent array dimensions."))
            _logpdf(d, x)
        end

        function pdf(d::MultivariateDistribution, x::AbstractVector)
            length(d) == length(x) ||
                throw(DimensionMismatch("Inconsistent array dimensions."))
            _pdf(d, x)
        end

    If there are better ways that can directly evaluate pdf values, one should override ``_pdf`` (*NOT* ``pdf``).

    The package also provides generic implementation of batch evaluation:

    .. code-block:: julia

        function _logpdf!(r::AbstractArray, d::MultivariateDistribution, x::DenseMatrix)
            for i = 1:size(x,2)
                @inbounds r[i] = _logpdf(d, view(x, :, i))
            end
            return r
        end

        function _pdf!(r::AbstractArray, d::MultivariateDistribution, x::DenseMatrix)
            for i = 1:size(x,2)
                @inbounds r[i] = _pdf(d, view(x, :, i))
            end
            return r
        end

        function logpdf!(r::AbstractArray, d::MultivariateDistribution, x::DenseMatrix)
            size(x) == (length(d), length(r)) ||
                throw(DimensionMismatch("Inconsistent array dimensions."))
            _logpdf!(r, d, x)
        end

        function pdf!(r::AbstractArray, d::MultivariateDistribution, x::DenseMatrix)
            size(x) == (length(d), length(r)) ||
                throw(DimensionMismatch("Inconsistent array dimensions."))
            _pdf!(r, d, x)
        end

        function logpdf(d::MultivariateDistribution, x::DenseMatrix)
            size(x,1) == length(d) ||
                throw(DimensionMismatch("Inconsistent array dimensions."))
            _logpdf!(Array(eltype(d), size(x,2)), d, x)
        end

        function pdf(d::MultivariateDistribution, x::DenseMatrix)
            size(x,1) == length(d) ||
                throw(DimensionMismatch("Inconsistent array dimensions."))
            _pdf!(Array(eltype(d), size(x,2)), d, x)
        end

    Note that if there exists faster methods for batch evaluation, one should override ``_logpdf!`` and ``_pdf!``.

It is also recommended that one also implements the following statistics functions: 

- ``mean``: compute the mean vector.
- ``var``:  compute the vector of element-wise variance. 
- ``entropy``: compute the entropy.
- ``cov``: compute the covariance matrix. (``cor`` is provided based on ``cov``).


Create a Matrix-variate Distribution
--------------------------------------

A multivariate distribution type should be defined as a subtype of ``DiscreteMatrixDistribution`` or ``ContinuousMatrixDistribution``. 

Following methods need to be implemented for each univariate distribution type (say ``D``):

.. function:: size(d::D)

    Return the size of each sample.

.. function:: _rand!{T<:Real}(d::D, x::AbstractMatrix{T})

    Generate a matrix sample to ``x``. 

    This function does not need to perform dimension checking. 

.. function:: sampler(d::D)

    Return a sampler for efficient batch/repeated sampling.

.. function:: _logpdf{T<:Real}(d::D, x::AbstractMatrix{T})

    Evaluate logarithm of pdf value for a given sample ``x``. This function need not perform dimension checking.


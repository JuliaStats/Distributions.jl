# Create New Samplers and Distributions

Whereas this package already provides a large collection of common distributions out of the box, there are still occasions where you want to create new distributions (*e.g.* your application requires a special kind of distribution, or you want to contribute to this package).

Generally, you don't have to implement every API method listed in the documentation. This package provides a series of generic functions that turn a small number of internal methods into user-end API methods. What you need to do is to implement this small set of internal methods for your distributions.

By default, `Discrete` sampleables have the support of type `Int` while `Continuous` sampleables have the support of type `Float64`. If this assumption does not hold for your new distribution or sampler, or its `ValueSupport` is neither `Discrete` nor `Continuous`, you should implement the `eltype` method in addition to the other methods listed below.

**Note:** The methods that need to be implemented are different for distributions of different variate forms.


## Create a Sampler

Unlike full-fledged distributions, a sampler, in general, only provides limited functionalities, mainly to support sampling.

### Univariate Sampler

To implement a univariate sampler, one can define a subtype (say `Spl`) of `Sampleable{Univariate,S}` (where `S` can be `Discrete` or `Continuous`), and provide a `rand` method, as

```julia
function rand(rng::AbstractRNG, s::Spl)
    # ... generate a single sample from s
end
```

The package already implements a vectorized version of `rand!` and `rand` that repeatedly calls the scalar version to generate multiple samples; as wells as a one arg version that uses the default random number generator.

### Multivariate Sampler

To implement a multivariate sampler, one can define a subtype of `Sampleable{Multivariate,S}`, and provide both `length` and `_rand!` methods, as

```julia
Base.length(s::Spl) = ... # return the length of each sample

function _rand!(rng::AbstractRNG, s::Spl, x::AbstractVector{T}) where T<:Real
    # ... generate a single vector sample to x
end
```

This function can assume that the dimension of `x` is correct, and doesn't need to perform dimension checking.

The package implements both `rand` and `rand!` as follows (which you don't need to implement in general):

```julia
function _rand!(rng::AbstractRNG, s::Sampleable{Multivariate}, A::DenseMatrix)
    for i = 1:size(A,2)
        _rand!(rng, s, view(A,:,i))
    end
    return A
end

function rand!(rng::AbstractRNG, s::Sampleable{Multivariate}, A::AbstractVector)
    length(A) == length(s) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    _rand!(rng, s, A)
end

function rand!(rng::AbstractRNG, s::Sampleable{Multivariate}, A::DenseMatrix)
    size(A,1) == length(s) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    _rand!(rng, s, A)
end

rand(rng::AbstractRNG, s::Sampleable{Multivariate,S}) where {S<:ValueSupport} =
    _rand!(rng, s, Vector{eltype(S)}(length(s)))

rand(rng::AbstractRNG, s::Sampleable{Multivariate,S}, n::Int) where {S<:ValueSupport} =
    _rand!(rng, s, Matrix{eltype(S)}(length(s), n))
```

If there is a more efficient method to generate multiple vector samples in a batch, one should provide the following method

```julia
function _rand!(rng::AbstractRNG, s::Spl, A::DenseMatrix{T}) where T<:Real
    # ... generate multiple vector samples in batch
end
```

Remember that each *column* of A is a sample.

### Matrix-variate Sampler

To implement a multivariate sampler, one can define a subtype of `Sampleable{Multivariate,S}`, and provide both `size` and `_rand!` methods, as

```julia
Base.size(s::Spl) = ... # the size of each matrix sample

function _rand!(rng::AbstractRNG, s::Spl, x::DenseMatrix{T}) where T<:Real
    # ... generate a single matrix sample to x
end
```

Note that you can assume `x` has correct dimensions in `_rand!` and don't have to perform dimension checking, the generic `rand` and `rand!` will do dimension checking and array allocation for you.

## Create a Distribution

Most distributions should implement a `sampler` method to improve batch sampling efficiency.

```@docs
sampler(d::Distribution)
```

### Univariate Distribution

A univariate distribution type should be defined as a subtype of `DiscreteUnivarateDistribution` or `ContinuousUnivariateDistribution`.

The following methods need to be implemented for each univariate distribution type:

- [`rand(::AbstractRNG, d::UnivariateDistribution)`](@ref)
- [`sampler(d::Distribution)`](@ref)
- [`logpdf(d::UnivariateDistribution, x::Real)`](@ref)
- [`cdf(d::UnivariateDistribution, x::Real)`](@ref)
- [`quantile(d::UnivariateDistribution, q::Real)`](@ref)
- [`minimum(d::UnivariateDistribution)`](@ref)
- [`maximum(d::UnivariateDistribution)`](@ref)
- [`insupport(d::UnivariateDistribution, x::Real)`](@ref)

It is also recommended that one also implements the following statistics functions:

- [`mean(d::UnivariateDistribution)`](@ref)
- [`var(d::UnivariateDistribution)`](@ref)
- [`modes(d::UnivariateDistribution)`](@ref)
- [`mode(d::UnivariateDistribution)`](@ref)
- [`skewness(d::UnivariateDistribution)`](@ref)
- [`kurtosis(d::Distribution, ::Bool)`](@ref)
- [`entropy(d::UnivariateDistribution, ::Real)`](@ref)
- [`mgf(d::UnivariateDistribution, ::Any)`](@ref)
- [`cf(d::UnivariateDistribution, ::Any)`](@ref)

You may refer to the source file `src/univariates.jl` to see details about how generic fallback functions for univariates are implemented.


## Create a Multivariate Distribution

A multivariate distribution type should be defined as a subtype of `DiscreteMultivarateDistribution` or `ContinuousMultivariateDistribution`.

The following methods need to be implemented for each multivariate distribution type:

- [`length(d::MultivariateDistribution)`](@ref)
- [`sampler(d::Distribution)`](@ref)
- [`eltype(d::Distribution)`](@ref)
- [`Distributions._rand!(::AbstractRNG, d::MultivariateDistribution, x::AbstractArray)`](@ref)
- [`Distributions._logpdf(d::MultivariateDistribution, x::AbstractArray)`](@ref)

Note that if there exist faster methods for batch evaluation, one should override `_logpdf!` and `_pdf!`.

Furthermore, the generic `loglikelihood` function repeatedly calls `_logpdf`. If there is
a better way to compute the log-likelihood, one should override `loglikelihood`.

It is also recommended that one also implements the following statistics functions:

- [`mean(d::MultivariateDistribution)`](@ref)
- [`var(d::MultivariateDistribution)`](@ref)
- [`entropy(d::MultivariateDistribution)`](@ref)
- [`cov(d::MultivariateDistribution)`](@ref)

## Create a Matrix-Variate Distribution

A matrix-variate distribution type should be defined as a subtype of `DiscreteMatrixDistribution` or `ContinuousMatrixDistribution`.

The following methods need to be implemented for each matrix-variate distribution type:

- [`size(d::MatrixDistribution)`](@ref)
- [`Distributions._rand!(rng::AbstractRNG, d::MatrixDistribution, A::AbstractMatrix)`](@ref)
- [`sampler(d::MatrixDistribution)`](@ref)
- [`Distributions._logpdf(d::MatrixDistribution, x::AbstractArray)`](@ref)

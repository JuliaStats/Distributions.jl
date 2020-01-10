# Create New Samplers and Distributions

Whereas this package already provides a large collection of common distributions out of box, there are still occasions where you want to create new distributions (*e.g* your application requires a special kind of distributions, or you want to contribute to this package).

Generally, you don't have to implement every API method listed in the documentation. This package provides a series of generic functions that turn a small number of internal methods into user-end API methods. What you need to do is to implement this small set of internal methods for your distributions.

By default, `Discrete` sampleables have support of type `Int` while `Continuous` sampleables have support of type `Float64`. If this assumption does not hold for your new distribution or sampler, or its `ValueSupport` is neither `Discrete` nor `Continuous`, you should implement the `eltype` method in addition to the other methods listed below.

**Note:** the methods need to be implemented are different for distributions of different variate forms.


## Create a Sampler

Unlike a full fledged distributions, a sampler, in general, only provides limited functionalities, mainly to support sampling.

### Univariate Sampler

To implement a univariate sampler, one can define a sub type (say `Spl`) of `Sampleable{Univariate,S}` (where `S` can be `Discrete` or `Continuous`), and provide a `rand` method, as

```julia
function rand(s::Spl)
    # ... generate a single sample from s
end
```

The package already implements a vectorized version of `rand!` and `rand` that repeatedly calls the he scalar version to generate multiple samples.

### Multivariate Sampler

To implement a multivariate sampler, one can define a sub type of `Sampleable{Multivariate,S}`, and provide both `length` and `_rand!` methods, as

```julia
Base.length(s::Spl) = ... # return the length of each sample

function _rand!(s::Spl, x::AbstractVector{T}) where T<:Real
    # ... generate a single vector sample to x
end
```

This function can assume that the dimension of `x` is correct, and doesn't need to perform dimension checking.

The package implements both `rand` and `rand!` as follows (which you don't need to implement in general):

```julia
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

rand(s::Sampleable{Multivariate,S}) where {S<:ValueSupport} =
    _rand!(s, Vector{eltype(S)}(length(s)))

rand(s::Sampleable{Multivariate,S}, n::Int) where {S<:ValueSupport} =
    _rand!(s, Matrix{eltype(S)}(length(s), n))
```

If there is a more efficient method to generate multiple vector samples in batch, one should provide the following method

```julia
function _rand!(s::Spl, A::DenseMatrix{T}) where T<:Real
    # ... generate multiple vector samples in batch
end
```

Remember that each *column* of A is a sample.

### Matrix-variate Sampler

To implement a multivariate sampler, one can define a sub type of `Sampleable{Multivariate,S}`, and provide both `size` and `_rand!` method, as

```julia
Base.size(s::Spl) = ... # the size of each matrix sample

function _rand!(s::Spl, x::DenseMatrix{T}) where T<:Real
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

Following methods need to be implemented for each univariate distribution type:

- [`rand(::AbstractRNG, d::UnivariateDistribution)`](@ref)
- [`sampler(d::Distribution)`](@ref)
- [`pdf(d::UnivariateDistribution, x::Real)`](@ref)
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

Following methods need to be implemented for each multivariate distribution type:

- [`length(d::MultivariateDistribution)`](@ref)
- [`sampler(d::Distribution)`](@ref)
- [`eltype(d::Distribution)`](@ref)
- [`Distributions._rand!(::AbstractRNG, d::MultivariateDistribution, x::AbstractArray)`](@ref)
- [`Distributions._logpdf(d::MultivariateDistribution, x::AbstractArray)`](@ref)

Note that if there exists faster methods for batch evaluation, one should override `_logpdf!` and `_pdf!`.

Furthermore, the generic `loglikelihood` function delegates to `_loglikelihood`, which repeatedly calls `_logpdf`. If there is a better way to compute log-likelihood, one should override `_loglikelihood`.

It is also recommended that one also implements the following statistics functions:

- [`mean(d::MultivariateDistribution)`](@ref)
- [`var(d::MultivariateDistribution)`](@ref)
- [`entropy(d::MultivariateDistribution)`](@ref)
- [`cov(d::MultivariateDistribution)`](@ref)

## Create a Matrix-variate Distribution

A multivariate distribution type should be defined as a subtype of `DiscreteMatrixDistribution` or `ContinuousMatrixDistribution`.

Following methods need to be implemented for each matrix-variate distribution type:

- [`size(d::MatrixDistribution)`](@ref)
- [`rand(d::MatrixDistribution)`](@ref)
- [`sampler(d::MatrixDistribution)`](@ref)
- [`Distributions._logpdf(d::MatrixDistribution, x::AbstractArray)`](@ref)

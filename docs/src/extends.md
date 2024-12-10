# Create New Samplers and Distributions

Whereas this package already provides a large collection of common distributions out of the box, there are still occasions where you want to create new distributions (*e.g.* your application requires a special kind of distribution, or you want to contribute to this package).

Generally, you don't have to implement every API method listed in the documentation. This package provides a series of generic functions that turn a small number of internal methods into user-end API methods. What you need to do is to implement this small set of internal methods for your distributions.

**Note:** The methods that need to be implemented are different for distributions of different variate forms.

## Create a Sampler

Unlike full-fledged distributions, a sampler, in general, only provides limited functionalities, mainly to support sampling.

### Univariate Sampler

To implement a univariate sampler, one can define a subtype (say `Spl`) of `Sampleable{Univariate,S}` (where `S` can be `Discrete` or `Continuous`), and provide a `rand` method, as

```julia
function Base.rand(rng::AbstractRNG, s::Spl)
    # ... generate a single sample from s
end
```

The package already implements vectorized versions `rand!(rng::AbstractRNG, s::Spl, dims::Int...)` and `rand(rng::AbstractRNG, s::Spl, dims::Int...)` that repeatedly call the scalar version to generate multiple samples.
Additionally, the package implements versions of these functions without the `rng::AbstractRNG` argument that use the default random number generator.

If there is a more efficient method to generate multiple samples, one should provide the following method

```julia
function Random.rand!(rng::AbstractRNG, s::Spl, x::AbstractArray{<:Real})
    # ... generate multiple samples from s in x
end
```

### Multivariate Sampler

To implement a multivariate sampler, one can define a subtype of `Sampleable{Multivariate,S}`, and provide `length`, `rand`, and `rand!` methods, as

```julia
Base.length(s::Spl) = ... # return the length of each sample

function Base.rand(rng::AbstractRNG, s::Spl)
    # ... generate a single vector sample from s
end

@inline function Random.rand!(rng::AbstractRNG, s::Spl, x::AbstractVector{<:Real})
    # `@inline` + `@boundscheck` allows users to skip bound checks by calling `@inbounds rand!(...)`
    # Ref https://docs.julialang.org/en/v1/devdocs/boundscheck/#Eliding-bounds-checks
    @boundscheck # ... check size (and possibly indices) of `x`
    # ... generate a single vector sample from s in x
end
```

If there is a more efficient method to generate multiple vector samples in a batch, one should provide the following method

```julia
@inline function Random.rand!(rng::AbstractRNG, s::Spl, A::AbstractMatrix{<:Real})
    # `@inline` + `@boundscheck` allows users to skip bound checks by calling `@inbounds rand!(...)`
    # Ref https://docs.julialang.org/en/v1/devdocs/boundscheck/#Eliding-bounds-checks
    @boundscheck # ... check size (and possibly indices) of `x`
    # ... generate multiple vector samples in batch
end
```

Remember that each *column* of A is a sample.

### Matrix-variate Sampler

To implement a multivariate sampler, one can define a subtype of `Sampleable{Multivariate,S}`, and provide `size`, `rand`, and `rand!` methods, as

```julia
Base.size(s::Spl) = ... # the size of each matrix sample

function Base.rand(rng::AbstractRNG, s::Spl)
    # ... generate a single matrix sample from s
end

@inline function Random.rand!(rng::AbstractRNG, s::Spl, x::AbstractMatrix{<:Real})
    # `@inline` + `@boundscheck` allows users to skip bound checks by calling `@inbounds rand!(...)`
    # Ref https://docs.julialang.org/en/v1/devdocs/boundscheck/#Eliding-bounds-checks
    @boundscheck # ... check size (and possibly indices) of `x`
    # ... generate a single matrix sample from s in x
end
```

## Create a Distribution

Most distributions should implement a `sampler` method to improve batch sampling efficiency.

```@docs
sampler(d::Distribution)
```

### Univariate Distribution

A univariate distribution type should be defined as a subtype of `DiscreteUnivarateDistribution` or `ContinuousUnivariateDistribution`.

The following methods need to be implemented for each univariate distribution type:

- [`Base.rand(::AbstractRNG, d::UnivariateDistribution)`](@ref)
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
- [`Base.rand(::AbstractRNG, d::MultivariateDistribution)`](@ref)
- [`Random.rand!(::AbstractRNG, d::MultivariateDistribution, x::AbstractVector{<:Real})`](@ref)
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
- [`Base.rand(rng::AbstractRNG, d::MatrixDistribution)`](@ref)
- [`Random.rand!(rng::AbstractRNG, d::MatrixDistribution, A::AbstractMatrix{<:Real})`](@ref)
- [`sampler(d::MatrixDistribution)`](@ref)
- [`Distributions._logpdf(d::MatrixDistribution, x::AbstractArray)`](@ref)

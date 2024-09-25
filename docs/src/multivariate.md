# [Multivariate Distributions](@id multivariates)

*Multivariate distributions* are the distributions whose variate forms are `Multivariate` (*i.e* each sample is a vector). Abstract types for multivariate distributions:

```julia
const MultivariateDistribution{S<:ValueSupport} = Distribution{Multivariate,S}

const DiscreteMultivariateDistribution   = Distribution{Multivariate, Discrete}
const ContinuousMultivariateDistribution = Distribution{Multivariate, Continuous}
```

## Common Interface

The methods listed below are implemented for each multivariate distribution, which provides a consistent interface to work with multivariate distributions.

### Computation of statistics

```@docs
length(::MultivariateDistribution)
size(::MultivariateDistribution)
eltype(::Type{MultivariateDistribution})
mean(::MultivariateDistribution)
var(::MultivariateDistribution)
cov(::MultivariateDistribution)
cor(::MultivariateDistribution)
entropy(::MultivariateDistribution)
entropy(::MultivariateDistribution, ::Real)
```

### Probability evaluation

```@docs
insupport(::MultivariateDistribution, ::AbstractArray)
pdf(::MultivariateDistribution, ::AbstractArray)
logpdf(::MultivariateDistribution, ::AbstractArray)
loglikelihood(::MultivariateDistribution, ::AbstractVector{<:Real})
```
**Note:** For multivariate distributions, the pdf value is usually very small or large, and therefore direct evaluation of the pdf may cause numerical problems. It is generally advisable to perform probability computation in log scale.


### Sampling

```@docs
rand(rng::AbstractRNG, ::MultivariateDistribution)
rand!(rng::AbstractRNG, d::MultivariateDistribution, x::AbstractArray)
```

**Note:** In addition to these common methods, each multivariate distribution has its special methods, as introduced below.


## Distributions

```@docs
Multinomial
Distributions.AbstractMvNormal
MvNormal
MvNormalCanon
MvLogitNormal
MvLogNormal
Dirichlet
```

## Addition Methods

### AbstractMvNormal

In addition to the methods listed in the common interface above, we also provide the following methods for all multivariate distributions under the base type `AbstractMvNormal`:

```@docs
invcov(::Distributions.AbstractMvNormal)
logdetcov(::Distributions.AbstractMvNormal)
sqmahal(::Distributions.AbstractMvNormal, ::AbstractArray)
rand(::AbstractRNG, ::Distributions.AbstractMvNormal)
minimum(::Distributions.AbstractMvNormal)
maximum(::Distributions.AbstractMvNormal)
extrema(::Distributions.AbstractMvNormal)
```


### MvLogNormal

In addition to the methods listed in the common interface above, we also provide the following methods:

```@docs
location(::MvLogNormal)
scale(::MvLogNormal)
median(::MvLogNormal)
mode(::MvLogNormal)
```

It can be necessary to calculate the parameters of the lognormal (location vector and scale matrix) from a given covariance and mean, median or mode. To that end, the following functions are provided.

```@docs
location{D<:Distributions.AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix)
location!{D<:Distributions.AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,μ::AbstractVector)
scale{D<:Distributions.AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix)
scale!{D<:Distributions.AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix)
params{D<:Distributions.AbstractMvLogNormal}(::Type{D},m::AbstractVector,S::AbstractMatrix)
```

## Internal Methods (for creating your own multivariate distribution)

```@docs
Distributions._logpdf(d::MultivariateDistribution, x::AbstractArray)
```

## Index

```@index
Pages = ["multivariate.md"]
```

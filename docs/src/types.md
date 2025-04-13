# Type Hierarchy

All samplers and distributions provided in this package are organized into a type hierarchy described as follows.

## Sampleable

The root of this type hierarchy is `Sampleable`. The abstract type `Sampleable` subsumes any types of objects from which one can draw samples, which particularly includes *samplers* and *distributions*. Formally, `Sampleable` is defined as

```julia
abstract type Sampleable{F<:VariateForm,S<:ValueSupport} end
```

It has two type parameters that define the kind of samples that can be drawn therefrom.

```@doc
Distributions.Sampleable
Base.rand(::Distributions.Sampleable)
```

### VariateForm

```@doc
Distributions.VariateForm
```

The `VariateForm` subtypes defined in `Distributions.jl` are:

**Type** | **A single sample** | **Multiple samples**
--- | --- |---
`Univariate == ArrayLikeVariate{0}` | a scalar number | A numeric array of arbitrary shape, each element being a sample
`Multivariate == ArrayLikeVariate{1}` | a numeric vector | A matrix, each column being a sample
`Matrixvariate == ArrayLikeVariate{2}` | a numeric matrix | An array of matrices, each element being a sample matrix

### ValueSupport

```@docs
Distributions.ValueSupport
```

The `ValueSupport` sub-types defined in `Distributions.jl` are:

```@docs
Distributions.Discrete
Distributions.Continuous
```

**Type** | **Default element type** | **Description** | **Examples**
--- | --- | --- | ---
`Discrete` | `Int` | Samples take countably many values | $\{0,1,2,3\}$, $\mathbb{N}$
`Continuous` | `Float64` | Samples take uncountably many values | $[0, 1]$, $\mathbb{R}$

Multiple samples are often organized into an array, depending on the variate form.

The basic functionalities that a sampleable object provides are to *retrieve information about the samples it generates* and to *draw samples*. Particularly, the following functions are provided for sampleable objects:

```@docs
length(::Sampleable)
size(::Sampleable)
nsamples(::Type{Sampleable}, ::Any)
eltype(::Type{Sampleable})
rand(::AbstractRNG, ::Sampleable)
rand!(::AbstractRNG, ::Sampleable, ::AbstractArray)
```

## Distributions

We use `Distribution`, a subtype of `Sampleable` as defined below, to capture probabilistic distributions. In addition to being sampleable, a *distribution* typically comes with an explicit way to combine its domain, probability density function, and many other quantities.

```julia
abstract type Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end
```

```@doc
Distributions.Distribution
```

To simplify the use in practice, we introduce a series of type alias as follows:
```julia
const UnivariateDistribution{S<:ValueSupport}   = Distribution{Univariate,S}
const MultivariateDistribution{S<:ValueSupport} = Distribution{Multivariate,S}
const MatrixDistribution{S<:ValueSupport}       = Distribution{Matrixvariate,S}
const NonMatrixDistribution = Union{UnivariateDistribution, MultivariateDistribution}

const DiscreteDistribution{F<:VariateForm}   = Distribution{F,Discrete}
const ContinuousDistribution{F<:VariateForm} = Distribution{F,Continuous}

const DiscreteUnivariateDistribution     = Distribution{Univariate,    Discrete}
const ContinuousUnivariateDistribution   = Distribution{Univariate,    Continuous}
const DiscreteMultivariateDistribution   = Distribution{Multivariate,  Discrete}
const ContinuousMultivariateDistribution = Distribution{Multivariate,  Continuous}
const DiscreteMatrixDistribution         = Distribution{Matrixvariate, Discrete}
const ContinuousMatrixDistribution       = Distribution{Matrixvariate, Continuous}
```

All methods applicable to `Sampleable` also apply to `Distribution`. The API for distributions of different variate forms are different (refer to [univariates](@ref univariates), [multivariates](@ref multivariates), and [matrix](@ref matrix-variates) for details).

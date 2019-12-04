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

The `VariateForm` sub-types defined in `Distributions.jl` are:

**Type** | **A single sample** | **Multiple samples**
--- | --- |---
`Univariate` | a scalar number | A numeric array of arbitrary shape, each element being a sample
`Multivariate` | a numeric vector | A matrix, each column being a sample
`Matrixvariate` | a numeric matrix | An array of matrices, each element being a sample matrix

### ValueSupport

```@doc
Distributions.ValueSupport
```

The `ValueSupport` sub-types defined in `Distributions.jl` are:

**Type** | **Element type** | **Descriptions**
--- | --- | ---
`CountableSupport{T}` | `T` | Samples take any discrete values
`ContiguousSupport{T <: Integer}` | `T` | Samples take any contiguous integer values
`Discrete = ContiguousSupport{Int}` | `Int` | Samples take `Int` values
`ContinuousSupport{T <: Number}` | `T` | Samples take continuous values
`Continuous = ContinuousSupport{Float64}` | `Float64` | Samples take continuous `Float64` values

Multiple samples are often organized into an array, depending on the variate form.

The basic functionalities that a sampleable object provides is to *retrieve information about the samples it generates* and to *draw samples*. Particularly, the following functions are provided for sampleable objects:

```@docs
length(::Sampleable)
size(::Sampleable)
nsamples(::Type{Sampleable}, ::Any)
eltype(::Type{Sampleable})
rand(::AbstractRNG, ::Sampleable)
rand!(::AbstractRNG, ::Sampleable, ::AbstractArray)
```

## Distributions

We use `Distribution`, a subtype of `Sampleable` as defined below, to capture probabilistic distributions. In addition to being sampleable, a *distribution* typically comes with an explicit way to combine its domain, probability density functions, among many other quantities.

```julia
abstract type Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end
```

```@doc
Distributions.Distribution
```

To simplify the use in practice, we introduce a series of type aliases as follows:
```julia
const UnivariateDistribution{S<:ValueSupport}   = Distribution{Univariate,S}
const MultivariateDistribution{S<:ValueSupport} = Distribution{Multivariate,S}
const MatrixDistribution{S<:ValueSupport}       = Distribution{Matrixvariate,S}

const CountableDistribution{F<:VariateForm, C<:CountableSupport} = Distribution{F,C}
const DiscreteDistribution{F<:VariateForm}                       = CountableDistribution{F,Discrete}
const ContinuousDistribution{F<:VariateForm}                     = Distribution{F,Continuous}

const CountableUnivariateDistribution{C<:CountableSupport} = UnivariateDistribution{C}
const DiscreteUnivariateDistribution                       = CountableUnivariateDistribution{Discrete}
const ContinuousUnivariateDistribution                     = UnivariateDistribution{Continuous}

const CountableMultivariateDistribution{C<:CountableSupport} = MultivariateDistribution{C}
const DiscreteMultivariateDistribution                       = CountableMultivariateDistribution{Discrete}
const ContinuousMultivariateDistribution                     = MultivariateDistribution{Continuous}

const CountableMatrixDistribution{C<:CountableSupport} = MatrixDistribution{C}
const DiscreteMatrixDistribution                       = CountableMatrixDistribution{Discrete}
const ContinuousMatrixDistribution                     = MatrixDistribution{Continuous}
```

All methods applicable to `Sampleable` also applies to `Distribution`. The API for distributions of different variate forms are different (refer to [univariates](@ref univariates), [multivariates](@ref multivariates), and [matrix](@ref matrix-variates) for details).

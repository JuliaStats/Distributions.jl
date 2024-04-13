# Mixture Models

A [mixture model](http://en.wikipedia.org/wiki/Mixture_model) is a probabilistic distribution that combines a set of *components* to represent the overall distribution. Generally, the probability density/mass function is given by a convex combination of the pdf/pmf of individual components, as

```math
f_{mix}(x; \Theta, \pi) = \sum_{k=1}^K \pi_k f(x; \theta_k)
```

A *mixture model* is characterized by a set of component parameters ``\Theta=\{\theta_1, \ldots, \theta_K\}`` and a prior distribution ``\pi`` over these components.


## Type Hierarchy

This package introduces a type `MixtureModel`, defined as follows, to represent a *mixture model*:

```julia
abstract type AbstractMixtureModel{VF<:VariateForm,VS<:ValueSupport} <: Distribution{VF, VS} end

struct MixtureModel{VF<:VariateForm,VS<:ValueSupport,Component<:Distribution} <: AbstractMixtureModel{VF,VS}
    components::Vector{Component}
    prior::Categorical
end

const UnivariateMixture    = AbstractMixtureModel{Univariate}
const MultivariateMixture  = AbstractMixtureModel{Multivariate}
```

**Remarks:**

- We introduce `AbstractMixtureModel` as a base type, which allows one to define a mixture model with different internal implementations, while still being able to leverage the common methods defined for `AbstractMixtureModel`.

```@docs
AbstractMixtureModel
```

- The `MixtureModel` is a parametric type, with three type parameters:

    - `VF`: the variate form, which can be `Univariate`, `Multivariate`, or `Matrixvariate`.
    - `VS`: the value support, which can be `Continuous` or `Discrete`.
    - `Component`: the type of component distributions, *e.g.* `Normal`.

- We define two aliases: `UnivariateMixture` and `MultivariateMixture`.

With such a type system, the type for a mixture of univariate normal distributions can be written as

```julia
MixtureModel{Univariate,Continuous,Normal}
```

## Constructors

```@docs
MixtureModel
```


**Examples**

```julia
# constructs a mixture of three normal distributions,
# with prior probabilities [0.2, 0.5, 0.3]
MixtureModel(Normal[
   Normal(-2.0, 1.2),
   Normal(0.0, 1.0),
   Normal(3.0, 2.5)], [0.2, 0.5, 0.3])

# if the components share the same prior, the prior vector can be omitted
MixtureModel(Normal[
   Normal(-2.0, 1.2),
   Normal(0.0, 1.0),
   Normal(3.0, 2.5)])

# Since all components have the same type, we can use a simplified syntax
MixtureModel(Normal, [(-2.0, 1.2), (0.0, 1.0), (3.0, 2.5)], [0.2, 0.5, 0.3])

# Again, one can omit the prior vector when all components share the same prior
MixtureModel(Normal, [(-2.0, 1.2), (0.0, 1.0), (3.0, 2.5)])

# The following example shows how one can make a Gaussian mixture
# where all components share the same unit variance
MixtureModel(map(u -> Normal(u, 1.0), [-2.0, 0.0, 3.0]))
```

## Common Interface

All subtypes of `AbstractMixtureModel` (obviously including `MixtureModel`) provide the following two methods:

```@docs
components(::AbstractMixtureModel)
probs(::AbstractMixtureModel)
Distributions.component_type(::AbstractMixtureModel)
```

In addition, for all subtypes of `UnivariateMixture` and `MultivariateMixture`, the following generic methods are provided:

```@docs
mean(::AbstractMixtureModel)
var(::UnivariateMixture)
length(::MultivariateMixture)
pdf(::AbstractMixtureModel, ::Any)
logpdf(::AbstractMixtureModel, ::Any)
rand(::AbstractMixtureModel)
rand!(::AbstractMixtureModel, ::AbstractArray)
```

## Estimation

There are several methods for the estimation of mixture models from data, and this problem remains an open research topic.
This package does not provide facilities for estimating mixture models. One can resort to other packages, *e.g.* [*GaussianMixtures.jl*](https://github.com/davidavdav/GaussianMixtures.jl), for this purpose.

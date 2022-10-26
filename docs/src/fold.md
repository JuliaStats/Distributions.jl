# Folded Distributions

A folded distribution ``F_{c}(D)`` of a distribution ``D`` at crease ``c \in \mathbb{R}`` is the reflection of the distribution below ``c`` onto the the distribution above ``c``. 

The pdf of such a distribution is given by:

```math
p(x | F_{c}(D)) = p(x | D) + p(x' | D)
```

where ``x' = 2c - x``


A folded distribution can be constructed using the following signature:

```@docs
folded
```

In the general case, this will create a `Folded{typeof(d)}`
structure, defined as follows:

```@docs
Folded
```

Many functions, including those for the evaluation of pdf and sampling,
are defined for all folded continuous univariate distributions:

- [`maximum(::ContinuousUnivariateDistribution)`](@ref)
- [`minimum(::ContinuousUnivariateDistribution)`](@ref)
- [`insupport(::ContinuousUnivariateDistribution, x::Any)`](@ref)
- [`pdf(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`logpdf(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`cdf(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`logcdf(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`logdiffcdf(::ContinuousUnivariateDistribution, ::T, ::T) where {T <: Real}`](@ref)
- [`ccdf(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`logccdf(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`quantile(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`cquantile(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`invlogcdf(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`invlogccdf(::ContinuousUnivariateDistribution, ::Real)`](@ref)
- [`rand(::ContinuousUnivariateDistribution)`](@ref)
- [`rand!(::ContinuousUnivariateDistribution, ::AbstractArray)`](@ref)
- [`median(::ContinuousUnivariateDistribution)`](@ref)

Functions to compute statistics, such as `mean`, `mode`, `var`, `std`, and `entropy`, are not available for generic folded distributions. The `mean` function *is* available for folded distributions when the `mean` function is defined on the corresponding `Truncated` distribution. 

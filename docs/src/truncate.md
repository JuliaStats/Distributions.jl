# Truncated Distributions

The package provides the `truncated` function which creates the most
appropriate distribution to represent a truncated version of a given
distribution.


A truncated distribution can be constructed using the following signature:

```@docs
truncated
```

In the general case, this will create a `Truncated{typeof(d)}`
structure, defined as follows:

```@docs
Truncated
```

Many functions, including those for the evaluation of pdf and sampling,
are defined for all truncated univariate distributions:

- [`maximum(::UnivariateDistribution)`](@ref)
- [`minimum(::UnivariateDistribution)`](@ref)
- [`insupport(::UnivariateDistribution, x::Any)`](@ref)
- [`pdf(::UnivariateDistribution, ::Real)`](@ref)
- [`logpdf(::UnivariateDistribution, ::Real)`](@ref)
- [`cdf(::UnivariateDistribution, ::Real)`](@ref)
- [`logcdf(::UnivariateDistribution, ::Real)`](@ref)
- [`logdiffcdf(::UnivariateDistribution, ::T, ::T) where {T <: Real}`](@ref)
- [`ccdf(::UnivariateDistribution, ::Real)`](@ref)
- [`logccdf(::UnivariateDistribution, ::Real)`](@ref)
- [`quantile(::UnivariateDistribution, ::Real)`](@ref)
- [`cquantile(::UnivariateDistribution, ::Real)`](@ref)
- [`invlogcdf(::UnivariateDistribution, ::Real)`](@ref)
- [`invlogccdf(::UnivariateDistribution, ::Real)`](@ref)
- [`rand(::UnivariateDistribution)`](@ref)
- [`rand!(::UnivariateDistribution, ::AbstractArray)`](@ref)
- [`median(::UnivariateDistribution)`](@ref)

Functions to compute statistics, such as `mean`, `mode`, `var`, `std`, and `entropy`, are not available for generic truncated distributions.
Generally, there are no easy ways to compute such quantities due to the complications incurred by truncation.
However, these methods are supported for truncated normal distributions `Truncated{<:Normal}` which can be constructed with `truncated(::Normal, ...)`.

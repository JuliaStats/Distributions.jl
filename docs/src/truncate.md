# Truncated Distributions

The package provides the `Truncated` type to represented truncated distributions, which is defined as below:

```julia
struct Truncated{D<:UnivariateDistribution,S<:ValueSupport} <: Distribution{Univariate,S}
    untruncated::D      # the original distribution (untruncated)
    lower::Float64      # lower bound
    upper::Float64      # upper bound
    lcdf::Float64       # cdf of lower bound
    ucdf::Float64       # cdf of upper bound

    tp::Float64         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::Float64      # log(tp), i.e. log(ucdf - lcdf)
end
```

A truncated distribution can be constructed using the constructor `Truncated` as follows:

```@docs
Truncated
```

Many functions, including those for the evaluation of pdf and sampling, are defined for all truncated univariate distributions:

- [`maximum(::UnivariateDistribution)`](@ref)
- [`minimum(::UnivariateDistribution)`](@ref)
- [`insupport(::UnivariateDistribution, x::Any)`](@ref)
- [`pdf(::UnivariateDistribution, ::Real)`](@ref)
- [`logpdf(::UnivariateDistribution, ::Real)`](@ref)
- [`cdf(::UnivariateDistribution, ::Real)`](@ref)
- [`logcdf(::UnivariateDistribution, ::Real)`](@ref)
- [`ccdf(::UnivariateDistribution, ::Real)`](@ref)
- [`logccdf(::UnivariateDistribution, ::Real)`](@ref)
- [`quantile(::UnivariateDistribution, ::Real)`](@ref)
- [`cquantile(::UnivariateDistribution, ::Real)`](@ref)
- [`invlogcdf(::UnivariateDistribution, ::Real)`](@ref)
- [`invlogccdf(::UnivariateDistribution, ::Real)`](@ref)
- [`rand(::UnivariateDistribution)`](@ref)
- [`rand!(::UnivariateDistribution, ::AbstractArray)`](@ref)
- [`median(::UnivariateDistribution)`](@ref)

Functions to compute statistics, such as `mean`, `mode`, `var`, `std`, and `entropy`, are not available for generic truncated distributions. Generally, there are no easy ways to compute such quantities due to the complications incurred by truncation.
However, these methods are supported for truncated normal distributions.

```@docs
TruncatedNormal
```

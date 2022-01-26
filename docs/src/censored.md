# Censored Distributions

In *censoring* of data, values exceeding an upper limit (right censoring) or falling below a lower limit (left censoring), or both (interval censoring) are replaced by the corresponding limit itself.
The package provides the `censored` function, which creates the most appropriate distribution to represent a censored version of a given distribution.

A censored distribution can be constructed using the following signature:

```@docs
censored
```

In the general case, this will create a `Distributions.Censored{typeof(d0)}` structure, defined as follows:

```@docs
Distributions.Censored
```

In general, `censored` should be called instead of the constructor of `Censored`, which is not exported.

Many functions, including those for the evaluation of pdf and sampling, are defined for all censored univariate distributions:

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
- [`median(::UnivariateDistribution)`](@ref)
- [`rand(::UnivariateDistribution)`](@ref)
- [`rand!(::UnivariateDistribution, ::AbstractArray)`](@ref)

Some functions to compute statistics are available for the censored distribution if they are also available for its truncation:
- [`mean(::UnivariateDistribution)`](@ref)
- [`var(::UnivariateDistribution)`](@ref)
- [`std(::UnivariateDistribution)`](@ref)
- [`entropy(::UnivariateDistribution)`](@ref)

For example, these functions are available for the following uncensored distributions:
- `DiscreteUniform`
- `Exponential`
- `LogUniform`
- `Normal`
- `Uniform`

[`mode`](@ref) is not implemented for censored distributions.

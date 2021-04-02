# Distribution Fitting to aggregate statistics

This package provides method to fit a distribution to a given
set of aggregate statistics.

```@meta
DocTestSetup = :(using Statistics,Distributions)
```
```jldoctest; output = false
# to specified moments
d = fit(LogNormal, Moments(3.0,4.0))
(mean(d), var(d)) .≈ (3.0, 4.0)

# to mean and upper quantile point
d = fit(LogNormal, 3, @qp_uu(8))
(mean(d), quantile(d, 0.975)) .≈ (3.0, 8.0)

# to mode and upper quantile point
d = fit(LogNormal, 3, @qp_uu(8), Val(:mode))
(mode(d), quantile(d, 0.975)) .≈ (3.0, 8.0)

# to two quantiles, i.e confidence range
d = fit(LogNormal, @qp_ll(1.0), @qp_uu(8))
(quantile(d, 0.025), quantile(d, 0.975)) .≈ (1.0, 8.0)

# approximate a different distribution by matching moments
dn = Normal(3,2)
d = fit(LogNormal, moments(dn))
(mean(d), var(d)) .≈ (3.0, 4.0)
# output
(true, true)
```

## Fit to statistical moments

```@docs
fit(::Type{D}, ::AbstractMoments) where {D<:Distribution}
```

```@docs
moments(d::Distribution, ::Val{N} = Val(2)) where N 
```

The syntax `Moments(mean,var)` produces an object of type `Moments <: AbstractMoments`.

```@docs
 AbstractMoments{N}
```

## Fit to several quantile points

```@docs
fit(::Type{D}, ::QuantilePoint, ::QuantilePoint) where {D<:Distribution}
```

## Fit to mean, mode, median and a quantile point

```@docs
fit(::Type{D}, ::Any, ::QuantilePoint, ::Val{stats} = Val(:mean)) where {D<:Distribution, stats}
```

## Implementing support for another distribution

In order to use the fitting framework for a distribution `MyDist`, one needs to implement the following four methods.

```julia
fit(::Type{MyDist}, m::AbstractMoments)

fit_mean_quantile(::Type{MyDist}, mean, qp::QuantilePoint)

fit_mode_quantile(::Type{MyDist}, mode, qp::QuantilePoint)

fit(::Type{MyDist}, lower::QuantilePoint, upper::QuantilePoint)
```

The default method for `fit` with `stats = :median` already works based on the methods for two quantile points.
If the general method on two quantile points cannot be specified, one can alternatively implement method:

```julia
fit_median_quantile(::Type{MyDist}, median, qp::QuantilePoint)
```







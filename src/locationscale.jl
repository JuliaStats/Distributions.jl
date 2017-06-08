immutable LocationScale{D<:UnivariateDistribution, S<:ValueSupport} <: UnivariateDistribution{S}
  base::D
  μ::Float64
  σ::Float64
end

### Constructors

function LocationScale(base::UnivariateDistribution, μ::Float64, σ::Float64)
  σ > 0 || throw(ArgumentError("Scale parameter σ should be positive"))
  LocationScale{typeof(base), value_support(typeof(base))}(base, μ, σ)
end

LocationScale(base::UnivariateDistribution, μ::Real, σ::Real) = LocationScale(base, Float64(μ), Float64(σ))

params(d::LocationScale) = tuple(params(d.base)..., d.μ, d.σ)
partype(d::LocationScale) = partype(d.base)

## range and support
islowerbounded(d::LocationScale) = islowerbounded(d.base)
isupperbounded(d::LocationScale) = isupperbounded(d.base)

minimum(d::LocationScale) = minimum(d.base)*d.σ + d.μ
maximum(d::LocationScale) = maximum(d.base)*d.σ + d.μ

insupport{D<:DiscreteUnivariateDistribution}(d::LocationScale{D, Discrete}, x::Real) =
  insupport(d.base, (x - d.μ)/d.σ)

## other common and useful functions

location(d::LocationScale) = d.μ
scale(d::LocationScale) = d.σ

pdf(d::LocationScale, x::Real) =
  pdf(d.base, (x - d.μ)/d.σ) / (typeof(d).parameters[2] == Continuous ? d.σ : 1)

logpdf(d::LocationScale, x::Real) =
  logpdf(d.base, (x - d.μ)/d.σ) - (typeof(d).parameters[2] == Continuous ? log(σ) : 0)

cdf(d::LocationScale, x::Real) =
   cdf(d.base, (x- d.μ) / d.σ)

logcdf(d::LocationScale, x::Real) =
   logcdf(d.base, (x - d.μ)/ d.σ)

quantile(d::LocationScale, x::Real) = quantile(d.base, x)*d.σ + d.μ

mean(d::LocationScale) = d.μ + d.σ*mean(d.base)
var(d::LocationScale) = var(d.base)*d.σ*d.σ
skewness(d::LocationScale) = skewness(d.base)
kurtosis(d::LocationScale) = kurtosis(d.base)

### Operations as convenience constructors

+(d::LocationScale, μ::Real) = LocationScale(d.base, d.μ + μ, d.σ)
-(d::LocationScale, μ::Real) = LocationScale(d.base, d.μ - μ, d.σ)
*(d::LocationScale, σ::Real) = LocationScale(d.base, d.μ*σ, d.σ*σ)
/(d::LocationScale, σ::Real) = LocationScale(d.base, d.μ/σ, d.σ/σ)


+(μ::Real, d::LocationScale) = d + μ
*(μ::Real, d::LocationScale) = d * μ

+(d::UnivariateDistribution, μ::Real) = LocationScale(d, μ, 1.0)
-(d::UnivariateDistribution, μ::Real) = LocationScale(d, -μ, 1.0)
*(d::UnivariateDistribution, σ::Real) = LocationScale(d, 0.0, σ)
/(d::UnivariateDistribution, σ::Real) = LocationScale(d, 0.0, 1/σ)


+(μ::Real, d::UnivariateDistribution) = d + μ
*(μ::Real, d::UnivariateDistribution) = d * μ

### Standardizing a LocationScale family
standardize(d::UnivariateDistribution) = (d- mean(d)) / std(d)

### Random sample generation

rand(d::LocationScale) = d.μ + d.σ*rand(d.base)

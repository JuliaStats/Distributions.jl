"""
    LocationScale(μ,σ,ρ)
    
A location-scale transformed distribution with location parameter `μ`,
scale parameter `σ`, and given distribution `ρ`.

```math
f(x) = \\frac{1}{σ} ρ \\! \\left( \\frac{x-μ}{σ} \\right)
```

```julia
LocationScale(μ,σ,ρ) # location-scale transformed distribution
params(d)            # Get the parameters, i.e. (μ, σ, and the base distribution)
location(d)          # Get the location parameter
scale(d)             # Get the scale parameter
```

External links
[Location-Scale family on Wikipedia](https://en.wikipedia.org/wiki/Location%E2%80%93scale_family)
"""
struct LocationScale{T<:Real, D<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ρ::D

    LocationScale{T}(μ::T,σ::T,ρ::ContinuousUnivariateDistribution) where T = (@check_args(LocationScale, σ > zero(σ)); new{T}(μ,σ,ρ))
end

LocationScale(μ::T,σ::T,ρ::ContinuousUnivariateDistribution) where {T<:Real} = LocationScale{T}(μ,σ,ρ)
LocationScale(μ::T,σ::T,ρ::ContinuousUnivariateDistribution) where {T<:Integer} = LocationScale{Float64}(Float64(μ),Float64(σ),ρ)

minimum(d::LocationScale{T}) where {T<:Real} = d.μ + d.σ * minimum(d.ρ)
maximum(d::LocationScale{T}) where {T<:Real} = d.μ + d.σ * maximum(d.ρ)

#### Conversions

convert(::Type{LocationScale{T}}, μ::Real, σ::Real, ρ::ContinuousUnivariateDistribution) where {T<:Real} = LocationScale(T(μ),T(σ),ρ)
convert(::Type{LocationScale{T}}, d::LocationScale{S}) where {T<:Real, S<:Real} = LocationScale(T(d.μ),T(d.σ),d.ρ)

#### Parameters

location(d::LocationScale{T}) where {T<:Real} = d.μ
scale(d::LocationScale{T}) where {T<:Real} = d.σ
params(d::LocationScale) = (d.μ,d.σ,d.ρ)
@inline partype(d::LocationScale{T}) where {T<:Real} = T

#### Statistics

mean(d::LocationScale{T}) where {T<:Real} = d.μ + d.σ * mean(d.ρ)
median(d::LocationScale{T}) where {T<:Real} = d.μ + d.σ * median(d.ρ)
mode(d::LocationScale{T}) where {T<:Real} = d.μ + d.σ * mode(d.ρ)
modes(d::LocationScale{T}) where {T<:Real} = d.μ + d.σ * modes(d.ρ)

var(d::LocationScale{T}) where {T<:Real} = d.σ^2 * var(d.ρ)
std(d::LocationScale{T}) where {T<:Real} = d.σ * std(d.ρ)
skewness(d::LocationScale{T}) where {T<:Real} = skewness(d.ρ)
kurtosis(d::LocationScale{T}) where {T<:Real} = kurtosis(d.ρ)

isplatykurtic(d::LocationScale{T}) where {T<:Real} = isplatykurtic(d.ρ)
isleptokurtic(d::LocationScale{T}) where {T<:Real} = isleptokurtic(d.ρ)
ismesokurtic(d::LocationScale{T}) where {T<:Real} = ismesokurtic(d.ρ)

entropy(d::LocationScale{T}) where {T<:Real} = entropy(d.ρ) + log(d.σ)
mgf(d::LocationScale{T},t::Real) where {T<:Real} = exp(d.μ*t) * mgf(d.ρ,d.σ*t)

#### Evaluation & Sampling

pdf(d::LocationScale{T},x::Real) where {T<:Real} = pdf(d.ρ,(x-d.μ)/d.σ) / d.σ
logpdf(d::LocationScale{T},x::Real) where {T<:Real} = logpdf(d.ρ,(x-d.μ)/d.σ) - log(d.σ)
cdf(d::LocationScale{T},x::Real) where {T<:Real} = cdf(d.ρ,(x-d.μ)/d.σ)
logcdf(d::LocationScale{T},x::Real) where {T<:Real} = logcdf(d.ρ,(x-d.μ)/d.σ)
quantile(d::LocationScale{T},q::Real) where {T<:Real} = d.μ + d.σ * quantile(d.ρ,q)

rand(d::LocationScale{T}) where {T<:Real} = d.μ + d.σ * rand(d.ρ)
cf(d::LocationScale{T}, t::Real) where {T<:Real} = cf(d.ρ,t*d.σ) * exp(1im*t*d.μ)
gradlogpdf(d::LocationScale{T}, x::Real) where {T<:Real} = gradlogpdf(d.ρ,(x-d.μ)/d.σ) / d.σ

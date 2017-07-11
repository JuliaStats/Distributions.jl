doc"""
    LocationScale(μ,σ,ρ)
    
A location-scale transformed distribution with location parameter `μ`,
scale parameter `σ`, and given distribution `ρ`.
$f(x) = \frac{1}{σ}ρ(\frac{x-μ}{σ})$

```julia
LocationScale(μ,σ,ρ) # location-scale transformed distribution
params(d)            # Get the parameters, i.e. (μ,σ,*****)
location(d)          # Get the location parameter
scale(d)             # Get the scale parameter
```

External links
[Location-Scale family on Wikipedia](https://en.wikipedia.org/wiki/Location%E2%80%93scale_family)
"""
immutable LocationScale{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ρ::ContinuousUnivariateDistribution

    (::Type{LocationScale{T}}){T}(μ::T,σ::T,ρ::ContinuousUnivariateDistribution) = (@check_args(LocationScale, σ > zero(σ)); new{T}(μ,σ,ρ))
end

LocationScale{T<:Real}(μ::T,σ::T,ρ::ContinuousUnivariateDistribution) = LocationScale{T}(μ,σ,ρ)
LocationScale{T<:Integer}(μ::T,σ::T,ρ::ContinuousUnivariateDistribution) = LocationScale{Float64}(Float64(μ),Float64(σ),ρ)

minimum{T<:Real}(d::LocationScale{T}) = d.μ + d.σ * minimum(d.ρ)
maximum{T<:Real}(d::LocationScale{T}) = d.μ + d.σ * maximum(d.ρ)

#### Conversions

convert{T<:Real}(::Type{LocationScale{T}}, μ::Real, σ::Real, ρ::ContinuousUnivariateDistribution) = LocationScale(T(μ),T(σ),ρ)
convert{T<:Real, S<:Real}(::Type{LocationScale{T}}, d::LocationScale{S}) = LocationScale(T(d.μ),T(d.σ),d.ρ)

#### Parameters

location{T<:Real}(d::LocationScale{T}) = d.μ
scale{T<:Real}(d::LocationScale{T}) = d.σ
params(d::LocationScale) = (d.μ,d.σ,d.ρ)
@inline partype{T<:Real}(d::LocationScale{T}) = T

#### Statistics

mean{T<:Real}(d::LocationScale{T}) = d.μ + d.σ * mean(d.ρ)
median{T<:Real}(d::LocationScale{T}) = d.μ + d.σ * median(d.ρ)
mode{T<:Real}(d::LocationScale{T}) = d.μ + d.σ * mode(d.ρ)
modes{T<:Real}(d::LocationScale{T}) = d.μ + d.σ * modes(d.ρ)

var{T<:Real}(d::LocationScale{T}) = d.σ^2 * var(d.ρ)
std{T<:Real}(d::LocationScale{T}) = d.σ * std(d.ρ)
skewness{T<:Real}(d::LocationScale{T}) = skewness(d.ρ)
kurtosis{T<:Real}(d::LocationScale{T}) = kurtosis(d.ρ)

isplatykurtic{T<:Real}(d::LocationScale{T}) = isplatykurtic(d.ρ)
isleptokurtic{T<:Real}(d::LocationScale{T}) = isleptokurtic(d.ρ)
ismesokurtic{T<:Real}(d::LocationScale{T}) = ismesokurtic(d.ρ)

entropy{T<:Real}(d::LocationScale{T}) = entropy(d.ρ) + log(d.σ)
mgf{T<:Real}(d::LocationScale{T},t::Real) = exp(d.μ*t) * mgf(d.ρ,d.σ*t)

#### Evaluation & Sampling

pdf{T<:Real}(d::LocationScale{T},x::Real) = pdf(d.ρ,(x-d.μ)/d.σ) / d.σ
logpdf{T<:Real}(d::LocationScale{T},x::Real) = logpdf(d.ρ,(x-d.μ)/d.σ) - log(d.σ)
cdf{T<:Real}(d::LocationScale{T},x::Real) = cdf(d.ρ,(x-d.μ)/d.σ)
logcdf{T<:Real}(d::LocationScale{T},x::Real) = logcdf(d.ρ,(x-d.μ)/d.σ)
quantile{T<:Real}(d::LocationScale{T},q::Real) = d.μ + d.σ * quantile(d.ρ,q)

rand{T<:Real}(d::LocationScale{T}) = d.μ + d.σ * rand(d.ρ)
cf{T<:Real}(d::LocationScale{T}, t::Real) = cf(d.ρ,t*d.σ) * exp(1im*t*d.μ)
gradlogpdf{T<:Real}(d::LocationScale{T}, x::Real) = gradlogpdf(d.ρ,(x-d.μ)/d.σ) / d.σ

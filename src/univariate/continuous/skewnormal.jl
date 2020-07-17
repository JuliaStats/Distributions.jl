"""
SkewNormal(ξ, ω, α)
    The *skew normal distribution* is a continuous probability distribution
    that generalises the normal distribution to allow for non-zero skewness.
External links
* [Skew normal distribution on Wikipedia](https://en.wikipedia.org/wiki/Skew_normal_distribution)
* [Discourse](https://discourse.julialang.org/t/skew-normal-distribution/21549/7)
* [SkewDist.jl](https://github.com/STOR-i/SkewDist.jl)
"""
struct SkewNormal{T<:Real} <: ContinuousUnivariateDistribution
    ξ::T
    ω::T
    α::T
    SkewNormal{T}(ξ::T, ω::T, α::T) where {T} = new{T}(ξ, ω, α)
end

function SkewNormal(ξ::T, ω::T, α::T; check_args=true) where {T <: Real}
    check_args && @check_args(SkewNormal, ω > zero(ω))
    return SkewNormal{T}(ξ, ω, α)
end

SkewNormal(ξ::Real, ω::Real, α::Real) = SkewNormal(promote(ξ, ω, α)...)
SkewNormal(ξ::Integer, ω::Integer, α::Integer) = SkewNormal(float(ξ), float(ω), float(α))
SkewNormal(α::T) where {T <: Real} = SkewNormal(zero(α), one(α), α)
SkewNormal() = SkewNormal(0.0, 1.0, 0.0)

@distr_support SkewNormal -Inf Inf

#### Conversions
convert(::Type{SkewNormal{T}}, ξ::S, ω::S, α::S) where {T <: Real, S <: Real} = SkewNormal(T(ξ), T(ω), T(α))
convert(::Type{SkewNormal{T}}, d::SkewNormal{S}) where {T <: Real, S <: Real} = SkewNormal(T(d.ξ), T(d.ω), T(d.α), check_args=false)

#### Parameters
params(d::SkewNormal) = (d.ξ, d.ω, d.α)
@inline partype(d::SkewNormal{T}) where {T<:Real} = T

#### Statistics
delta(d::SkewNormal) = d.α / √(1 + d.α^2)
mean_z(d::SkewNormal) = √(2/π) * delta(d)
std_z(d::SkewNormal) = 1 - (2/π) * delta(d)^2

mean(d::SkewNormal) = d.ξ + d.ω * mean_z(d)
var(d::SkewNormal) = abs2(d.ω) * (1 - mean_z(d)^2)
std(d::SkewNormal) = √var(d)
skewness(d::SkewNormal) = ((4 - π)/2) * (mean_z(d)^3/(1 - mean_z(d)^2)^(3/2))
kurtosis(d::SkewNormal) = 2 * (π-3) * ((delta(d) * sqrt(2/π))^4/(1-2 * (delta(d)^2)/π)^2) 

# no analytic expression for max m_0(d) but accurate numerical approximation 
m_0(d::SkewNormal) = mean_z(d) - (skewness(d) * std_z(d))/2 - (sign(d.α)/2) * exp(-2π/abs(d.α))
mode(d::SkewNormal) = d.ξ + d.ω * m_0(d)  

#### Evalution
pdf(d::SkewNormal, x::Real) = (2/d.ω) * normpdf((x-d.ξ)/d.ω) * normcdf(d.α * (x-d.ξ)/d.ω)
logpdf(d::SkewNormal, x::Real) = log(2) - log(d.ω) + normlogpdf((x-d.ξ) / d.ω) + normlogcdf(d.α * (x-d.ξ) / d.ω)
#cdf requires Owen's T function.
#cdf/quantile etc 

mgf(d::SkewNormal, t::Real) = 2 * exp(d.ξ * t + (d.ω^2 * t^2)/2 ) * normcdf(d.ω * delta(d) * t)

cf(d::SkewNormal, t::Real) = exp(im * t * d.ξ - (d.ω^2 * t^2)/2) * (1 + im * erfi((d.ω * delta(d) * t)/(sqrt(2))) )

#### Sampling
function rand(rng::AbstractRNG, d::SkewNormal)
    u0 = randn(rng)
    v = randn(rng)
    δ = delta(d)
    u1 = δ * u0 + √(1-δ^2) * v
    return d.ξ + d.ω * sign(u0) * u1
end

## Fitting  # to be added see: https://github.com/STOR-i/SkewDist.jl/issues/3



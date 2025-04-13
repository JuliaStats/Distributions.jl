"""
    SkewNormal(ξ, ω, α)

The *skew normal distribution* is a continuous probability distribution that
generalises the normal distribution to allow for non-zero skewness. Given a
location `ξ`, scale `ω`, and shape `α`, it has the probability density function

```math
f(x; \\xi, \\omega, \\alpha) =
\\frac{2}{\\omega \\sqrt{2 \\pi}} \\exp{\\bigg(-\\frac{(x-\\xi)^2}{2\\omega^2}\\bigg)}
\\int_{-\\infty}^{\\alpha\\left(\\frac{x-\\xi}{\\omega}\\right)}
\\frac{1}{\\sqrt{2 \\pi}}  \\exp{\\bigg(-\\frac{t^2}{2}\\bigg)} \\, \\mathrm{d}t
```

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

function SkewNormal(ξ::T, ω::T, α::T; check_args::Bool=true) where {T <: Real}
    @check_args SkewNormal (ω, ω > zero(ω))
    return SkewNormal{T}(ξ, ω, α)
end

SkewNormal(ξ::Real, ω::Real, α::Real; check_args::Bool=true) = SkewNormal(promote(ξ, ω, α)...; check_args=check_args)
SkewNormal(ξ::Integer, ω::Integer, α::Integer; check_args::Bool=true) = SkewNormal(float(ξ), float(ω), float(α); check_args=check_args)
SkewNormal(α::Real=0.0) = SkewNormal(zero(α), one(α), α; check_args=false)

@distr_support SkewNormal -Inf Inf

#### Conversions
convert(::Type{SkewNormal{T}}, ξ::S, ω::S, α::S) where {T <: Real, S <: Real} = SkewNormal(T(ξ), T(ω), T(α))
Base.convert(::Type{SkewNormal{T}}, d::SkewNormal) where {T<:Real} = SkewNormal{T}(T(d.ξ), T(d.ω), T(d.α))
Base.convert(::Type{SkewNormal{T}}, d::SkewNormal{T}) where {T<:Real} = d

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

#### Evaluation
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



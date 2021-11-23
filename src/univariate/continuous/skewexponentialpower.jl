"""
    SkewExponentialPower(μ, σ, p, α)

The *Skewed exponential power distribution*, with location `μ`, scale `σ`, shape `p`, and skewness `α`
has the probability density function [1]
```math
f(x; \\mu, \\sigma, p, \\alpha) =
\\begin{cases}
\\frac{1}{\\sigma 2p^{1/p}\\Gamma(1+1/p)} \\exp \\left\\{ - \\frac{1}{2p}\\Big| \\frac{x-\\mu}{\\alpha \\sigma} \\Big|^p \\right\\}, & \\text{if } x \\leq \\mu \\\\
\\frac{1}{\\sigma 2p^{1/p}\\Gamma(1+1/p)} \\exp \\left\\{ - \\frac{1}{2p}\\Big| \\frac{x-\\mu}{(1-\\alpha) \\sigma} \\Big|^p \\right\\}, & \\text{if } x > \\mu
\\end{cases}.
```
The Skewed exponential power distribution (SEPD) incorporates the laplace (``p=1, \\alpha=0.5``),
normal (``p=2, \\alpha=0.5``), uniform (``p\\rightarrow \\infty, \\alpha=0.5``), asymmetric laplace (``p=1``), skew normal (``p=2``),
and exponential power distribution (``\\alpha = 0.5``) as special cases.

[1] Zhy, D. and V. Zinde-Walsh (2009). Properties and estimation of asymmetric exponential power distribution. _Journal of econometrics_, 148(1):86-96, 2009.

```julia
SkewExponentialPower()            # SEPD with shape 2, scale 1, location 0, and skewness 0.5 (the standard normal distribution)
SkewExponentialPower(μ, σ, p, α)  # SEPD with location μ, scale σ, shape p, and skewness α
SkewExponentialPower(μ, σ, p)     # SEPD with location μ, scale σ, shape p, and skewness 0.5 (the exponential power distribution)
SkewExponentialPower(μ, σ)        # SEPD with location μ, scale σ, shape 2, and skewness 0.5 (the normal distribution)
SkewExponentialPower(μ)           # SEPD with location μ, scale 1, shape 2, and skewness 0.5 (the normal distribution)

params(d)       # Get the parameters, i.e. (μ, σ, p, α)
shape(d)        # Get the shape parameter, i.e. p
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. σ
```
"""
struct SkewExponentialPower{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    p::T
    α::T
    SkewExponentialPower{T}(μ::T, σ::T, p::T, α::T) where {T} = new{T}(μ, σ, p, α)
end

function SkewExponentialPower(µ::T, σ::T, p::T, α::T; check_args=true) where {T <: Real}
    check_args && @check_args(SkewExponentialPower, σ > zero(σ))
    check_args && @check_args(SkewExponentialPower, p > zero(p))
    check_args && @check_args(SkewExponentialPower, α > zero(α) && α < one(α))
    return SkewExponentialPower{T}(µ, σ, p, α)
end

SkewExponentialPower(μ::Real, σ::Real, p::Real, α::Real) = SkewExponentialPower(promote(μ, σ, p, α)...)
SkewExponentialPower(μ::Real, σ::Real, p::Real) = SkewExponentialPower(promote(μ, σ, p, 0.5)...)
SkewExponentialPower(μ::Real, σ::Real) = SkewExponentialPower(promote(μ, σ, 2., 0.5)...)
SkewExponentialPower(μ::Real) = SkewExponentialPower(promote(μ, 1., 2., 0.5)...)
SkewExponentialPower() = SkewExponentialPower(0., 1., 2., 0.5)

@distr_support SkewExponentialPower -Inf Inf

### Conversions
convert(::Type{SkewExponentialPower{T}}, μ::S, σ::S, p::S, α::S) where {T <: Real, S <: Real} = SkewExponentialPower(T(μ), T(σ), T(p), T(α))
convert(::Type{SkewExponentialPower{T}}, d::SkewExponentialPower{S}) where {T <: Real, S <: Real} = SkewExponentialPower(T(d.μ), T(d.σ), T(d.p), T(d.α), check_args=false)

### Parameters
@inline partype(d::SkewExponentialPower{T}) where {T<:Real} = T

params(d::SkewExponentialPower) = (d.μ, d.σ, d.p, d.α)
location(d::SkewExponentialPower) = d.μ
shape(d::SkewExponentialPower) = d.p
scale(d::SkewExponentialPower) = d.σ

### Statistics

#Calculates the kth central moment of the SEPD, see Equation 18 in [1].
function m_k(d::SkewExponentialPower, k::Integer)
    _, σ, p, α = params(d)
    (2*p^(1/p))^k*((-1)^k*α^(1+k) + (1-α)^(1+k)) *
        σ^k * gamma((1+k)/p) / gamma(1/p)
end

mean(d::SkewExponentialPower) = m_k(d, 1) + d.μ
mode(d::SkewExponentialPower) =  m_k(d, 1) + d.μ
var(d::SkewExponentialPower) = m_k(d, 2) - m_k(d, 1)^2
std(d::SkewExponentialPower) = √var(d)
skewness(d::SkewExponentialPower) = m_k(d, 3)/(std(d))^3
kurtosis(d::SkewExponentialPower) = m_k(d, 4)/var(d)^2 - 3.

function logpdf(d::SkewExponentialPower, x::Real)
    μ, σ, p, α = params(d)
    -log(σ) - log(2*p^(1/p)*gamma(1+1/p)) - 1/p*(x < μ ? ((μ-x)/(2*σ*α))^p : ((x-μ)/(2*σ*(1-α)))^p)
end

pdf(d::SkewExponentialPower, x::Real) = exp(logpdf(d, x))

function cdf(d::SkewExponentialPower, x::Real)
    μ, σ, p, α = params(d)
    if x <= μ
        α * (1 - cdf(Gamma(1/p), 1/p * (abs((x-μ)/σ) / (2*α))^p))
    else
        α + (1-α) * cdf(Gamma(1/p), 1/p * (abs((x-μ)/σ) / (2*(1-α)))^p)
    end
end

function quantile(d::SkewExponentialPower, p::Real)
    μ, σ, p, α = params(d)
    if q <= α
        μ - 2*α*σ * (p * quantile(Gamma(1/p), 1-q/α))^(1/p)
    else
        μ + 2*(1-α)*σ * (p * quantile(Gamma(1/p), 1-(1-q)/(1-α)))^(1/p)
    end
end

function rand(rng::AbstractRNG, d::SkewExponentialPower)
    μ, σ, p, α = params(d)
    if rand(rng) < d.α
        μ - σ * 2*p^(1/p) * α * rand(Gamma(1/p, 1))^(1/p)
    else
        μ + σ * 2*p^(1/p) * (1-α) * rand(Gamma(1/p, 1))^(1/p)
    end
end

"""
    SkewedExponentialPower(μ, σ, p, α)

The *Skewed exponential power distribution*, with location `μ`, scale `σ`, shape `p`, and skewness `α`,
has the probability density function [1]
```math
f(x; \\mu, \\sigma, p, \\alpha) =
\\begin{cases}
\\frac{1}{\\sigma 2p^{1/p}\\Gamma(1+1/p)} \\exp \\left\\{ - \\frac{1}{2p}\\Big| \\frac{x-\\mu}{\\alpha \\sigma} \\Big|^p \\right\\}, & \\text{if } x \\leq \\mu \\\\
\\frac{1}{\\sigma 2p^{1/p}\\Gamma(1+1/p)} \\exp \\left\\{ - \\frac{1}{2p}\\Big| \\frac{x-\\mu}{(1-\\alpha) \\sigma} \\Big|^p \\right\\}, & \\text{if } x > \\mu
\\end{cases}.
```
The Skewed exponential power distribution (SEPD) incorporates the Laplace (``p=1, \\alpha=0.5``),
normal (``p=2, \\alpha=0.5``), uniform (``p\\rightarrow \\infty, \\alpha=0.5``), asymmetric Laplace (``p=1``), skew normal (``p=2``),
and exponential power distribution (``\\alpha = 0.5``) as special cases.

[1] Zhy, D. and V. Zinde-Walsh (2009). Properties and estimation of asymmetric exponential power distribution. _Journal of econometrics_, 148(1):86-96, 2009.

```julia
SkewedExponentialPower()            # SEPD with shape 2, scale 1, location 0, and skewness 0.5 (the standard normal distribution)
SkewedExponentialPower(μ, σ, p, α)  # SEPD with location μ, scale σ, shape p, and skewness α
SkewedExponentialPower(μ, σ, p)     # SEPD with location μ, scale σ, shape p, and skewness 0.5 (the exponential power distribution)
SkewedExponentialPower(μ, σ)        # SEPD with location μ, scale σ, shape 2, and skewness 0.5 (the normal distribution)
SkewedExponentialPower(μ)           # SEPD with location μ, scale 1, shape 2, and skewness 0.5 (the normal distribution)

params(d)       # Get the parameters, i.e. (μ, σ, p, α)
shape(d)        # Get the shape parameter, i.e. p
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. σ
```
"""
struct SkewedExponentialPower{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    p::T
    α::T
    SkewedExponentialPower{T}(μ::T, σ::T, p::T, α::T) where {T} = new{T}(μ, σ, p, α)
end

function SkewedExponentialPower(µ::T, σ::T, p::T, α::T; check_args::Bool=true) where {T <: Real}
    @check_args SkewedExponentialPower (σ, σ > zero(σ)) (p, p > zero(p)) (α, zero(α) < α < one(α))
    return SkewedExponentialPower{T}(µ, σ, p, α)
end

function SkewedExponentialPower(μ::Real, σ::Real, p::Real=2, α::Real=1//2; check_args::Bool=true)
    return SkewedExponentialPower(promote(μ, σ, p, α)...; check_args=check_args)
end
SkewedExponentialPower(μ::Real=0) = SkewedExponentialPower(μ, 1, 2, 1//2; check_args=false)

@distr_support SkewedExponentialPower -Inf Inf

### Conversions
function Base.convert(::Type{SkewedExponentialPower{T}}, d::SkewedExponentialPower) where {T<:Real}
    SkewedExponentialPower{T}(T(d.μ), T(d.σ), T(d.p), T(d.α))
end
Base.convert(::Type{SkewedExponentialPower{T}}, d::SkewedExponentialPower{T}) where {T<:Real} = d

### Parameters
@inline partype(::SkewedExponentialPower{T}) where {T<:Real} = T

params(d::SkewedExponentialPower) = (d.μ, d.σ, d.p, d.α)
location(d::SkewedExponentialPower) = d.μ
shape(d::SkewedExponentialPower) = d.p
scale(d::SkewedExponentialPower) = d.σ

### Statistics

#Calculates the kth central moment of the SEPD
function m_k(d::SkewedExponentialPower, k::Integer)
    _, σ, p, α = params(d)
    inv_p = inv(p)
    return  k * (logtwo + inv_p * log(p) + log(σ)) + loggamma((1 + k) * inv_p) -
        loggamma(inv_p) + log(abs((-1)^k * α^(1 + k) + (1 - α)^(1 + k)))
end

# needed for odd moments in log scale
sgn(d::SkewedExponentialPower) = d.α > 1//2 ? -1 : 1

mean(d::SkewedExponentialPower) = d.α == 1//2 ? float(d.μ) : sgn(d)*exp(m_k(d, 1)) + d.μ
mode(d::SkewedExponentialPower) =  mean(d)
var(d::SkewedExponentialPower) = exp(m_k(d, 2)) - exp(2*m_k(d, 1))
skewness(d::SkewedExponentialPower) = d.α == 1//2 ? float(zero(partype(d))) : sgn(d)*exp(m_k(d, 3)) / (std(d))^3
kurtosis(d::SkewedExponentialPower) = exp(m_k(d, 4))/var(d)^2 - 3

function logpdf(d::SkewedExponentialPower, x::Real)
    μ, σ, p, α = params(d)
    a = x < μ ? α : 1 - α
    inv_p = inv(p)
    return -(logtwo + log(σ) + loggamma(inv_p) + ((1 - p) * log(p) + (abs(μ - x) / (2 * σ * a))^p) / p)
end

function cdf(d::SkewedExponentialPower, x::Real)
    μ, σ, p, α = params(d)
    inv_p = inv(p)
    if x <= μ
        α * ccdf(Gamma(inv_p), inv_p * (abs((x-μ)/σ) / (2*α))^p)
    else
        α + (1-α) * cdf(Gamma(inv_p), inv_p * (abs((x-μ)/σ) / (2*(1-α)))^p)
    end
end
function logcdf(d::SkewedExponentialPower, x::Real)
    μ, σ, p, α = params(d)
    inv_p = inv(p)
    if x <= μ
        log(α) + logccdf(Gamma(inv_p), inv_p * (abs((x-μ)/σ) / (2*α))^p)
    else
        log1mexp(log1p(-α) + logccdf(Gamma(inv_p), inv_p * (abs((x-μ)/σ) / (2*(1-α)))^p))
    end
end

function quantile(d::SkewedExponentialPower, p::Real)
    μ, σ, _, α = params(d)
    inv_p = inv(d.p)
    if p <= α
        μ - 2*α*σ * (d.p * quantile(Gamma(inv_p), (α-p)/α))^inv_p
    else
        μ + 2*(1-α)*σ * (d.p * quantile(Gamma(inv_p), (p-α)/(1-α)))^inv_p
    end
end

function rand(rng::AbstractRNG, d::SkewedExponentialPower)
    μ, σ, p, α = params(d)
    inv_p = inv(d.p)
    z = 2*σ * (p * rand(rng, Gamma(inv_p, 1)))^inv_p
    if rand(rng) < d.α
        return μ - α * z
    else
        return μ + (1-α) * z
    end
end

"""
    AsymmetricExponentialPower(μ, σ, p₁, p₂, α)

The *Asymmetric exponential power distribution*, with location `μ`, scale `σ`, left (`x < μ`) shape `p₁`, right (`x >= μ`) shape `p₂`, and skewness `α`,
has the probability density function [1]
```math
f(x; \\mu, \\sigma, p_1, p_2, \\alpha) =
\\begin{cases}
\\frac{\\alpha}{\\alpha^*} \\frac{1}{\\sigma} K_{EP}(p_1) \\exp \\left\\{ - \\frac{1}{2p_1}\\Big| \\frac{x-\\mu}{\\alpha^* \\sigma} \\Big|^{p_1} \\right\\}, & \\text{if } x \\leq \\mu \\\\
\\frac{(1-\\alpha)}{(1-\\alpha^*)} \\frac{1}{\\sigma} K_{EP}(p_2) \\exp \\left\\{ - \\frac{1}{2p_2}\\Big| \\frac{x-\\mu}{(1-\\alpha^*) \\sigma} \\Big|^{p_2} \\right\\}, & \\text{if } x > \\mu
\\end{cases},
```
Where ``K_{EP}(p) = 1/(2p^{1/p}\\Gamma(1+1/p_1))`` and
```math
\\alpha^* = \\frac{\\alpha K_{EP}(p_1)}{\\alpha K_{EP}(p_1) + (1-\\alpha) K_{EP}(p_2)}
```
The asymmetric exponential power distribution (AEPD) the skewed exponential power distribution as special case (``p_1 = p_2``) and thus the 
Laplace, Normal, uniform, exponential power distribution, asymmetric Laplace and skew normal are also special cases. 

[1] Zhy, D. and V. Zinde-Walsh (2009). Properties and estimation of asymmetric exponential power distribution. _Journal of econometrics_, 148(1):86-96, 2009.

```julia
AsymmetricExponentialPower()                # AEPD with location 0, scale 1 left shape 2, right shape 2, and skewness 0.5 (the standard normal distribution)
AsymmetricExponentialPower(μ, σ, p₁, p₂ α)  # AEPD with location μ, scale σ, left shape p₁, right shape p₂, and skewness α
AsymmetricExponentialPower(μ, σ, p₁, p₂)    # AEPD with location μ, scale σ, left shape p₁, right shape p₂, and skewness 0.5 (the exponential power distribution)
AsymmetricExponentialPower(μ, σ)            # AEPD with location μ, scale σ, left shape 2, right shape 2, and skewness 0.5 (the normal distribution)
AsymmetricExponentialPower(μ)               # AEPD with location μ, scale 1, left shape 2, right shape 2, and skewness 0.5 (the normal distribution)

params(d)       # Get the parameters, i.e. (μ, σ, p₁, p₂, α)
shape(d)        # Get the shape parameters, i.e. (p₁, p₂)
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. σ
```
"""
struct AsymmetricExponentialPower{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    p₁::T
    p₂::T
    α::T
    AsymmetricExponentialPower{T}(μ::T, σ::T, p₁::T, p₂::T, α::T) where {T} = new{T}(μ, σ, p₁, p₂, α)
end


function AsymmetricExponentialPower(µ::T, σ::T, p₁::T, p₂::T, α::T; check_args::Bool=true) where {T <: Real}
    @check_args AsymmetricExponentialPower (σ, σ > zero(σ)) (p₁, p₁ > zero(p₁)) (p₂, p₂ > zero(p₂)) (α, zero(α) < α < one(α))
    return AsymmetricExponentialPower{T}(µ, σ, p₁, p₂, α)
end

function AsymmetricExponentialPower(μ::Real, σ::Real, p₁::Real, p₂::Real, α::Real = 1//2; check_args::Bool=true)
    return AsymmetricExponentialPower(promote(μ, σ, p₁, p₂, α)...; check_args=check_args)
end

function AsymmetricExponentialPower(μ::Real, σ::Real; check_args::Bool=true)
    return AsymmetricExponentialPower(promote(μ, σ, 2, 2, 1//2)...; check_args=check_args)
end

AsymmetricExponentialPower(μ::Real=0) = AsymmetricExponentialPower(μ, 1, 2, 2, 1//2; check_args=false)

@distr_support AsymmetricExponentialPower -Inf Inf

### Conversions
function Base.convert(::Type{AsymmetricExponentialPower{T}}, d::AsymmetricExponentialPower) where {T<:Real}
    AsymmetricExponentialPower{T}(T(d.μ), T(d.σ), T(d.p), T(d.α))
end
Base.convert(::Type{AsymmetricExponentialPower{T}}, d::AsymmetricExponentialPower{T}) where {T<:Real} = d

### Parameters
@inline partype(::AsymmetricExponentialPower{T}) where {T<:Real} = T
params(d::AsymmetricExponentialPower) = (d.μ, d.σ, d.p₁, d.p₂, d.α)
location(d::AsymmetricExponentialPower) = d.μ
shape(d::AsymmetricExponentialPower) = (d.p₁, d.p₂)
scale(d::AsymmetricExponentialPower) = d.σ

# log of Equation (4) of [1]

### Statistics
# Computes log K_{EP}(p), Zhy, D. and V. Zinde-Walsh (2009)
function logK(p::Real)
    inv_p = inv(p)
    return -(logtwo + loggamma(inv_p) + ((1 - p) * log(p))/p)
end

# Equation (3) in Zhy, D. and V. Zinde-Walsh (2009)
function αstar(α::Real, p₁::Real, p₂::Real)
    K1 = exp(logK(p₁))
    K2 = exp(logK(p₂))
    return α*K1 / (α*K1 + (1-α)*K2)
end

# Computes Equation 4 in Zhy, D. and V. Zinde-Walsh (2009)
B(α::Real, p₁::Real, p₂::Real) = α*exp(logK(p₁)) + (1-α)*exp(logK(p₂))

#Calculates the kth central moment of the AEPD, Equation 14 in Zhy, D. and V. Zinde-Walsh (2009)
function m_k(d::AsymmetricExponentialPower, k::Integer)
    _, σ, p₁, p₂, α = params(d)
    inv_p1, inv_p2 = inv(p₁), inv(p₂)
    H1 = k*log(p₁) + loggamma((1+k)*inv_p1) - (1+k)*loggamma(inv_p1)
    H2 = k*log(p₂) + loggamma((1+k)*inv_p2) - (1+k)*loggamma(inv_p2)
    return B(α, p₁, p₂)^(-k) * σ^k * ((-1)^k * α^(1+k)*exp(H1) + (1-α)^(1+k)*exp(H2))
end

mean(d::AsymmetricExponentialPower) = d.α == 1//2 ? float(d.μ) : m_k(d, 1) + d.μ
var(d::AsymmetricExponentialPower) = m_k(d, 2) - m_k(d, 1)^2
skewness(d::AsymmetricExponentialPower) = d.α == 1//2 ? float(zero(partype(d))) : m_k(d, 3) / (std(d))^3
kurtosis(d::AsymmetricExponentialPower) = m_k(d, 4)/var(d)^2 - 3

function logpdf(d::AsymmetricExponentialPower, x::Real)
    μ, σ, p₁, p₂, α = params(d)
    a, astar, inv_p, p = x < μ ? (α, αstar(α, p₁, p₂), inv(p₁), p₁) : (1 - α, 1-αstar(α, p₁, p₂), inv(p₂), p₂)
    return -(log(astar) - log(a) + logtwo + log(σ) + loggamma(inv_p) + ((1 - p) * log(p) + (abs(μ - x) / (2 * σ * astar))^p) / p)
end

function cdf(d::AsymmetricExponentialPower, x::Real)
    μ, σ, p₁, p₂, α = params(d)
    if x <= μ
        inv_p = inv(p₁)
        α * ccdf(Gamma(inv_p), inv_p * (abs((x-μ)/σ) / (2*αstar(α, p₁, p₂)))^p₁)
    else
        inv_p = inv(p₂)
        α + (1-α) * cdf(Gamma(inv_p), inv_p * (abs((x-μ)/σ) / (2*(1-αstar(α, p₁, p₂))))^p₂)
    end
end

function quantile(d::AsymmetricExponentialPower, p::Real)
    μ, σ, p₁, p₂, α = params(d)
    if p <= α
        inv_p = inv(p₁)
        μ - 2*αstar(α, p₁, p₂)*σ * (p₁ * quantile(Gamma(inv_p), (α-p)/α))^inv_p
    else
        inv_p = inv(p₂)
        μ + 2*(1-αstar(α, p₁, p₂))*σ * (p₂ * quantile(Gamma(inv_p), (p-α)/(1-α)))^inv_p
    end
end

function rand(rng::AbstractRNG, d::AsymmetricExponentialPower)
    μ, σ, p₁, p₂, α = params(d)
    if rand(rng) < α
        inv_p = inv(p₁)
        z = 2*σ * (p₁ * rand(rng,Gamma(inv_p, 1)))^inv_p
        return μ - αstar(α,p₁,p₂) * z
    else
        inv_p = inv(p₂)
        z = 2*σ * (p₂ * rand(rng,Gamma(inv_p, 1)))^inv_p
        return μ + (1-αstar(α,p₁,p₂)) * z
    end
end
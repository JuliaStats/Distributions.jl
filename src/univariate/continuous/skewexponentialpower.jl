"""
    SkewExponentialPower(μ, σ, p, α)

The *Skewed exponential power distribution*, with location `μ`, scale `σ`, shape `p`, and skewness `α`
has the probability density function
```math
f(x; \\mu, \\sigma, \\p, \\alpha) =
\\begin{cases}
\\frac{1}{\\sigma}K_{EP}(p_1) \\exp \\left\\{ - \\frac{1}{2p}\\Big| \\frac{x-\\mu}{\\alpha \\sigma} \\Big|^{p_1} \\right\\}, & \\text{if } x \\leq \\mu; \\\\
\\frac{1}{\\sigma}K_{EP}(p_2) \\exp \\left\\{ - \\frac{1}{2p}\\Big| \\frac{x-\\mu}{(1-\\alpha) \\sigma} \\Big|^{p_2} \\right\\}, & \\text{if } x > \\mu,
\\end{cases}
```
where ``K_{EP}(p) = 1/[2p^{1/p}\\Gamma(1+1/p)]``.
The Skewed exponential power distribution (SEPD) incorporates the laplace (``p=1, \\alpha=0.5``),
normal (``p=2, \\alpha=0.5``), uniform (``p\\rightarrow \\infty, \\alpha=0.5``), asymmetric laplace (``p=1``), skew normal (``p=2``),
and exponential power distribution (``\\alpha = 0.5``) as special cases.

```julia
SkewExponentialPower()            # SEPD with shape 2, scale 1, location 0, and skewness 0.5 (the standard normal distribution)
SkewExponentialPower(μ, σ, p, α)  # SEPD with location μ, scale σ, shape p, and skewness α
SkewExponentialPower(μ, σ, p)     # SEPD with location μ, scale σ, shape p, and skewness 0.5 (the exponential power distribution)
SkewExponentialPower(μ, σ)        # SEPD with location μ, scale σ, shape 2, and skewness 0.5 (the normal distribution)
SkewExponentialPower(σ)           # SEPD with location 0, scale σ, shape 2, and skewness 0.5 (the normal distribution)

params(d)       # Get the parameters, i.e. (μ, σ, p, α)
shape(d)        # Get the shape parameter, p
skewness(d)     # Get the skewness parameter, α
location(d)     # Get the location parameter, μ
```
"""
struct SkewedExponentialPower{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    p::T
    α::T
    SkewedExponentialPower{T}(μ::T, σ::T, p::T, α::T) where {T} = new{T}(μ, σ, p, α)
end

function SkewedExponentialPower(µ::T, σ::T, p::T, α::T; check_args=true) where {T <: Real}
    check_args && @check_args(SkewedExponentialPower, σ > zero(σ))
    check_args && @check_args(SkewedExponentialPower, p > zero(p))
    check_args && @check_args(SkewedExponentialPower, α > zero(α) && α < one(α))
    return SkewedExponentialPower{T}(µ, σ, p, α)
end

SkewedExponentialPower(μ::Real, σ::Real, p::Real, α::Real) = SkewedExponentialPower(promote(μ, σ, p, α)...)
SkewedExponentialPower(μ::Real, σ::Real, p::Real) = SkewedExponentialPower(promote(μ, σ, p, 0.5)...)
SkewedExponentialPower(μ::Real, σ::Real) = SkewedExponentialPower(promote(μ, σ, 2., 0.5)...)
SkewedExponentialPower(σ::Real) = SkewedExponentialPower(promote(0., σ, 2., 0.5)...)

@distr_support SkewedExponentialPower -Inf Inf

### Conversions
convert(::Type{SkewedExponentialPower{T}}, μ::S, σ::S, p::S, α::S) where {T <: Real, S <: Real} = SkewedExponentialPower(T(μ), T(σ), T(p), T(α))
convert(::Type{SkewedExponentialPower{T}}, d::SkewedExponentialPower{S}) where {T <: Real, S <: Real} = SkewedExponentialPower((T(d.μ), T(d.σ), T(d.p), T(d.α), check_args=false)

### Parameters
@inline partype(d::SkewedExponentialPower{T}) where {T<:Real} = T

params(d::SkewedExponentialPower) = (d.μ, d.σ, d.p, d.α)
location(d::SkewedExponentialPower) = d.μ
shape(d::SkewedExponentialPower) = d.p
scale(d::SkewedExponentialPower) = d.σ
skew(d::SkewedExponentialPower) = d.α

### Statistics
"""
    m_k(d, k)

Calculates the kth central moment of the SEPD, see Equation 18 in [1].
[1]
"""
function m_k(d::SkewedExponentialPower, k::Integer)
    (2*d.p^(1/d.p))^2*((-1)^k*d.α^(1+k) + (1-α)^(1+k)) *
    σ * a * gamma((1+k)/d.p) / gamma(1/p)
end
mean(d::SkewedExponentialPower) = m_k(d, 1) + d.μ
median(d::SkewedExponentialPower) = d.μ
mode(d::SkewedExponentialPower) =  m_k(d, 1) + d.μ
var(d::SkewedExponentialPower) = m_k(d, 2) - m_k(d, 1)^2
skewness(d::SkewedExponentialPower) = m_k(d, 3)
kurtosis(d::SkewedExponentialPower) = m_k(d, 4)

function logpdf(d::SkewedExponentialPower, x::Real)
    μ, σ, p, α = params(d)
    -log(σ) - log(2*p^(1/p)*gamma(1+1/p)) - 1/p*(x < μ ? ((μ-x)/(2*σ*α))^p : ((x-μ)/(2*σ*(1-α)))^p)
end

pdf(d::SkewedExponentialPower, x::Real) = exp(logpdf(d, x))

function rand(rng::AbstractRNG, d::SkewedExponentialPower)
    μ, σ, p, α = params(d)
    if rand(rng) < d.α
        μ - σ * 2*p^(1/p) * α * rand(Gamma(1/p, 1))^(1/p)
    else
        μ + σ * 2*p^(1/p) * (1-α) * rand(Gamma(1/p, 1))^(1/p)
    end
end

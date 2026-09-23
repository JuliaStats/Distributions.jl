"""
    NormalInverseGaussian(μ,α,β,δ)

The *Normal-inverse Gaussian distribution* with location `μ`, tail heaviness `α`, asymmetry parameter `β` and scale `δ` has probability density function

```math
f(x; \\mu, \\alpha, \\beta, \\delta) = \\frac{\\alpha\\delta K_1 \\left(\\alpha\\sqrt{\\delta^2 + (x - \\mu)^2}\\right)}{\\pi \\sqrt{\\delta^2 + (x - \\mu)^2}} \\; e^{\\delta \\gamma + \\beta (x - \\mu)}
```
where ``K_j`` denotes a modified Bessel function of the third kind.


External links

* [Normal-inverse Gaussian distribution on Wikipedia](http://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution)

"""
struct NormalInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    α::T
    β::T
    δ::T
    γ::T
    function NormalInverseGaussian{T}(μ::T, α::T, β::T, δ::T) where T
        γ = sqrt(α^2 - β^2)

        new{T}(μ, α, β, δ, γ)
    end
end

NormalInverseGaussian(μ::T, α::T, β::T, δ::T) where {T<:Real} = NormalInverseGaussian{T}(μ, α, β, δ)
NormalInverseGaussian(μ::Real, α::Real, β::Real, δ::Real) = NormalInverseGaussian(promote(μ, α, β, δ)...)
function NormalInverseGaussian(μ::Integer, α::Integer, β::Integer, δ::Integer)
    return NormalInverseGaussian(float(μ), float(α), float(β), float(δ))
end

@distr_support NormalInverseGaussian -Inf Inf

#### Conversions
function convert(::Type{NormalInverseGaussian{T}}, μ::Real, α::Real, β::Real, δ::Real) where T<:Real
    NormalInverseGaussian(T(μ), T(α), T(β), T(δ))
end
function Base.convert(::Type{NormalInverseGaussian{T}}, d::NormalInverseGaussian) where {T<:Real}
    NormalInverseGaussian{T}(T(d.μ), T(d.α), T(d.β), T(d.δ))
end
Base.convert(::Type{NormalInverseGaussian{T}}, d::NormalInverseGaussian{T}) where {T<:Real} = d

params(d::NormalInverseGaussian) = (d.μ, d.α, d.β, d.δ)
partype(::Type{<:NormalInverseGaussian{T}}) where {T<:Real} = T

mean(d::NormalInverseGaussian) = d.μ + d.δ * d.β / d.γ
var(d::NormalInverseGaussian) = d.δ * d.α^2 / d.γ^3
skewness(d::NormalInverseGaussian) = 3d.β / (d.α * sqrt(d.δ * d.γ))
kurtosis(d::NormalInverseGaussian) = 3 * (1 + 4*d.β^2/d.α^2) / (d.δ * d.γ)

function pdf(d::NormalInverseGaussian, x::Real)
    μ, α, β, δ = params(d)
    α * δ * besselk(1, α*sqrt(δ^2+(x - μ)^2)) / (π*sqrt(δ^2 + (x - μ)^2)) * exp(δ * d.γ + β*(x - μ))
end

function logpdf(d::NormalInverseGaussian, x::Real)
    μ, α, β, δ = params(d)
    log(α*δ) + log(besselk(1, α*sqrt(δ^2+(x-μ)^2))) - log(π*sqrt(δ^2+(x-μ)^2)) + δ*d.γ + β*(x-μ)
end


#### Sampling

# The Normal Inverse Gaussian distribution is a normal variance-mean
# mixture with an inverse Gaussian as mixing distribution.
#
# Ole E. Barndorff-Nielsen (1997)
# Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling
# Scandinavian Journal of Statistics, Vol. 24, pp. 1--13
# DOI: http://dx.doi.org/10.1111/1467-9469.00045

function rand(rng::Random.AbstractRNG, d::NormalInverseGaussian)
    μ, α, β, δ = params(d)

    Z = InverseGaussian(δ/d.γ, δ^2)
    z = rand(rng, Z)
    X = Normal(μ + β*z, sqrt(z))
    return rand(rng, X)
end

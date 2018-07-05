"""
    Gumbel(μ, θ)

The *Gumbel distribution*  with location `μ` and scale `θ` has probability density function

```math
f(x; \\mu, \\theta) = \\frac{1}{\\theta} e^{-(z + e^z)},
\\quad \\text{ with } z = \\frac{x - \\mu}{\\theta}
```

```julia
Gumbel()            # Gumbel distribution with zero location and unit scale, i.e. Gumbel(0, 1)
Gumbel(u)           # Gumbel distribution with location u and unit scale, i.e. Gumbel(u, 1)
Gumbel(u, b)        # Gumbel distribution with location u and scale b

params(d)        # Get the parameters, i.e. (u, b)
location(d)      # Get the location parameter, i.e. u
scale(d)         # Get the scale parameter, i.e. b
```

External links

* [Gumbel distribution on Wikipedia](http://en.wikipedia.org/wiki/Gumbel_distribution)
"""
struct Gumbel{T<:Real} <: ContinuousUnivariateDistribution
    μ::T  # location
    θ::T  # scale

    Gumbel{T}(μ::T, θ::T) where {T} = (@check_args(Gumbel, θ > zero(θ)); new{T}(μ, θ))
end

Gumbel(μ::T, θ::T) where {T<:Real} = Gumbel{T}(μ, θ)
Gumbel(μ::Real, θ::Real) = Gumbel(promote(μ, θ)...)
Gumbel(μ::Integer, θ::Integer) = Gumbel(Float64(μ), Float64(θ))
Gumbel(μ::Real) = Gumbel(μ, 1.0)
Gumbel() = Gumbel(0.0, 1.0)

@distr_support Gumbel -Inf Inf

const DoubleExponential = Gumbel

#### Conversions

convert(::Type{Gumbel{T}}, μ::S, θ::S) where {T <: Real, S <: Real} = Gumbel(T(μ), T(θ))
convert(::Type{Gumbel{T}}, d::Gumbel{S}) where {T <: Real, S <: Real} = Gumbel(T(d.μ), T(d.θ))

#### Parameters

location(d::Gumbel) = d.μ
scale(d::Gumbel) = d.θ
params(d::Gumbel) = (d.μ, d.θ)
@inline partype(d::Gumbel{T}) where {T<:Real} = T


#### Statistics

mean(d::Gumbel) = d.μ + d.θ * MathConstants.γ

median(d::Gumbel{T}) where {T<:Real} = d.μ - d.θ * log(T(logtwo))

mode(d::Gumbel) = d.μ

var(d::Gumbel{T}) where {T<:Real} = T(π)^2/6 * d.θ^2

skewness(d::Gumbel{T}) where {T<:Real} = 12*sqrt(T(6))*zeta(T(3)) / π^3

kurtosis(d::Gumbel{T}) where {T<:Real} = T(12)/5

entropy(d::Gumbel) = log(d.θ) + 1 + MathConstants.γ


#### Evaluation

zval(d::Gumbel, x::Real) = (x - d.μ) / d.θ
xval(d::Gumbel, z::Real) = x * d.θ + d.μ

function pdf(d::Gumbel, x::Real)
    z = zval(d, x)
    exp(-z - exp(-z)) / d.θ
end

function logpdf(d::Gumbel, x::Real)
    z = zval(d, x)
    - (z + exp(-z) + log(d.θ))
end

cdf(d::Gumbel, x::Real) = exp(-exp(-zval(d, x)))
logcdf(d::Gumbel, x::Real) = -exp(-zval(d, x))

quantile(d::Gumbel, p::Real) = d.μ - d.θ * log(-log(p))

gradlogpdf(d::Gumbel, x::Real) = - (1 + exp((d.μ - x) / d.θ)) / d.θ


#### Sampling

rand(d::Gumbel) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Gumbel) = quantile(d, rand(rng))

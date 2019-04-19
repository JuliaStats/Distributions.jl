"""
    Gumbel <: ContinuousUnivariateDistribution

The *Gumbel* or *double exponential* probability distribution.

# Constructors

    Gumbel(μ|mu|location=0, θ|theta|scale=1)

Construct a Gumbel distribution with location parameter `μ` and scale `θ`.

    Gumbel(mean=,std=)
    Gumbel(mean=,var=)

Construct a Gumbel distribution matching the relevant moments.

# Details

The Gumbel distribution  with location `μ` and scale `θ` has probability density function

```math
f(x; \\mu, \\theta) = \\frac{1}{\\theta} e^{-(z + e^-z)},
\\quad \\text{ with } z = \\frac{x - \\mu}{\\theta}
```

# Examples

```julia
Gumbel()
Gumbel(μ=2,θ=3)
```

# External links

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

@kwdispatch (::Type{D})(;mu=>μ, location=>μ, theta=>θ, scale=>θ) where {D<:Gumbel} begin
    () -> D(0,1)
    (μ) -> D(μ,1)
    (θ) -> D(0,θ)
    (μ,θ) -> D(μ,θ)

    function (mean, std)
        θ = sqrt(6)*std/π
        μ = mean - θ * MathConstants.γ
        D(μ, θ)
    end
    function (mean, var)
        θ = sqrt(6*var)/π
        μ = mean - θ * MathConstants.γ
        D(μ, θ)
    end
end

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

"""
    Gumbel(μ, θ)

The *Gumbel (maxima) distribution*  with location `μ` and scale `θ` has probability density function

```math
f(x; \\mu, \\theta) = \\frac{1}{\\theta} e^{-(z + e^{-z})},
\\quad \\text{ with } z = \\frac{x - \\mu}{\\theta}
```

```julia
Gumbel()            # Gumbel distribution with zero location and unit scale, i.e. Gumbel(0, 1)
Gumbel(μ)           # Gumbel distribution with location μ and unit scale, i.e. Gumbel(μ, 1)
Gumbel(μ, θ)        # Gumbel distribution with location μ and scale θ

params(d)        # Get the parameters, i.e. (μ, θ)
location(d)      # Get the location parameter, i.e. μ
scale(d)         # Get the scale parameter, i.e. θ
```

External links

* [Gumbel distribution on Wikipedia](http://en.wikipedia.org/wiki/Gumbel_distribution)
"""
struct Gumbel{T<:Real} <: ContinuousUnivariateDistribution
    μ::T  # location
    θ::T  # scale
    Gumbel{T}(µ::T, θ::T) where {T} = new{T}(µ, θ)
end

function Gumbel(μ::T, θ::T; check_args::Bool=true) where {T <: Real}
    @check_args Gumbel (θ, θ > zero(θ))
    return Gumbel{T}(μ, θ)
end

Gumbel(μ::Real, θ::Real; check_args::Bool=true) = Gumbel(promote(μ, θ)...; check_args=check_args)
Gumbel(μ::Integer, θ::Integer; check_args::Bool=true) = Gumbel(float(μ), float(θ); check_args=check_args)
Gumbel(μ::Real=0.0) = Gumbel(μ, one(μ); check_args=false)

@distr_support Gumbel -Inf Inf

const DoubleExponential = Gumbel

Base.eltype(::Type{Gumbel{T}}) where {T} = T

#### Conversions

convert(::Type{Gumbel{T}}, μ::S, θ::S) where {T <: Real, S <: Real} = Gumbel(T(μ), T(θ))
Base.convert(::Type{Gumbel{T}}, d::Gumbel) where {T<:Real} = Gumbel{T}(T(d.μ), T(d.θ))
Base.convert(::Type{Gumbel{T}}, d::Gumbel{T}) where {T<:Real} = d

#### Parameters

location(d::Gumbel) = d.μ
scale(d::Gumbel) = d.θ
params(d::Gumbel) = (d.μ, d.θ)
partype(::Gumbel{T}) where {T} = T

function Base.rand(rng::Random.AbstractRNG, d::Gumbel)
    return d.μ - d.θ * log(randexp(rng, float(eltype(d))))
end
function _rand!(rng::Random.AbstractRNG, d::Gumbel, x::AbstractArray{<:Real})
    randexp!(rng, x)
    x .= d.μ .- d.θ .* log.(x)
    return x
end

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
xval(d::Gumbel, z::Real) = z * d.θ + d.μ

function pdf(d::Gumbel, x::Real)
    z = zval(d, x)
    exp(-z - exp(-z)) / d.θ
end

function logpdf(d::Gumbel, x::Real)
    z = zval(d, x)
    - (z + exp(-z) + log(d.θ))
end

cdf(d::Gumbel, x::Real) = exp(-exp(-zval(d, x)))
ccdf(d::Gumbel, x::Real) = -expm1(-exp(-zval(d, x)))
logcdf(d::Gumbel, x::Real) = -exp(-zval(d, x))
logccdf(d::Gumbel, x::Real) = log1mexp(-exp(-zval(d, x)))

quantile(d::Gumbel, p::Real) = xval(d, -log(-log(p)))
cquantile(d::Gumbel, p::Real) = xval(d, -log(-log1p(-p)))
invlogcdf(d::Gumbel, lp::Real) = xval(d, -log(-lp))
invlogccdf(d::Gumbel, lp::Real) = xval(d, -log(-log1mexp(lp)))

gradlogpdf(d::Gumbel, x::Real) = expm1(-zval(d, x)) / d.θ

mgf(d::Gumbel, t::Real) = gamma(1 - d.θ * t) * exp(d.μ * t)
cgf(d::Gumbel, t::Real) = loggamma(1 - d.θ * t) + d.μ * t
cf(d::Gumbel, t::Real) = gamma(1 - im * d.θ * t) * cis(d.μ * t)

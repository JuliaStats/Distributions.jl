"""
    InverseGamma(α, θ)

The *inverse Gamma distribution* with shape parameter `α` and scale `θ` has probability
density function

```math
f(x; \\alpha, \\theta) = \\frac{\\theta^\\alpha x^{-(\\alpha + 1)}}{\\Gamma(\\alpha)}
e^{-\\frac{\\theta}{x}}, \\quad x > 0
```

It is related to the [`Gamma`](@ref) distribution: if ``X \\sim \\operatorname{Gamma}(\\alpha, \\beta)``, then ``1 / X \\sim \\operatorname{InverseGamma}(\\alpha, \\beta^{-1})``.

```julia
InverseGamma()        # Inverse Gamma distribution with unit shape and unit scale, i.e. InverseGamma(1, 1)
InverseGamma(α)       # Inverse Gamma distribution with shape α and unit scale, i.e. InverseGamma(α, 1)
InverseGamma(α, θ)    # Inverse Gamma distribution with shape α and scale θ

params(d)        # Get the parameters, i.e. (α, θ)
shape(d)         # Get the shape parameter, i.e. α
scale(d)         # Get the scale parameter, i.e. θ
```

External links

* [Inverse gamma distribution on Wikipedia](http://en.wikipedia.org/wiki/Inverse-gamma_distribution)
"""
struct InverseGamma{T<:Real} <: ContinuousUnivariateDistribution
    invd::Gamma{T}
    θ::T
    InverseGamma{T}(α::T, θ::T) where {T<:Real} = new{T}(Gamma(α, inv(θ), check_args=false), θ)
end

function InverseGamma(α::T, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(InverseGamma, α > zero(α) && θ > zero(θ))
    return InverseGamma{T}(α, θ)
end

InverseGamma(α::Real, θ::Real) = InverseGamma(promote(α, θ)...)
InverseGamma(α::Integer, θ::Integer) = InverseGamma(float(α), float(θ))
InverseGamma(α::T) where {T <: Real} = InverseGamma(α, one(T))
InverseGamma() = InverseGamma(1.0, 1.0, check_args=false)

@distr_support InverseGamma 0.0 Inf

#### Conversions
convert(::Type{InverseGamma{T}}, α::S, θ::S) where {T <: Real, S <: Real} = InverseGamma(T(α), T(θ))
convert(::Type{InverseGamma{T}}, d::InverseGamma{S}) where {T <: Real, S <: Real} = InverseGamma(T(shape(d.invd)), T(d.θ))

#### Parameters

shape(d::InverseGamma) = shape(d.invd)
scale(d::InverseGamma) = d.θ
rate(d::InverseGamma) = scale(d.invd)

params(d::InverseGamma) = (shape(d), scale(d))
partype(::InverseGamma{T}) where {T} = T


#### Parameters

mean(d::InverseGamma{T}) where {T} = ((α, θ) = params(d); α  > 1 ? θ / (α - 1) : T(Inf))

mode(d::InverseGamma) = scale(d) / (shape(d) + 1)

function var(d::InverseGamma{T}) where T<:Real
    (α, θ) = params(d)
    α > 2 ? θ^2 / ((α - 1)^2 * (α - 2)) : T(Inf)
end

function skewness(d::InverseGamma{T}) where T<:Real
    α = shape(d)
    α > 3 ? 4sqrt(α - 2) / (α - 3) : T(NaN)
end

function kurtosis(d::InverseGamma{T}) where T<:Real
    α = shape(d)
    α > 4 ? (30α - 66) / ((α - 3) * (α - 4)) : T(NaN)
end

function entropy(d::InverseGamma)
    (α, θ) = params(d)
    α + loggamma(α) - (1 + α) * digamma(α) + log(θ)
end


#### Evaluation

pdf(d::InverseGamma, x::Real) = exp(logpdf(d, x))

function logpdf(d::InverseGamma, x::Real)
    (α, θ) = params(d)
    α * log(θ) - loggamma(α) - (α + 1) * log(x) - θ / x
end

cdf(d::InverseGamma, x::Real) = ccdf(d.invd, 1 / x)
ccdf(d::InverseGamma, x::Real) = cdf(d.invd, 1 / x)
logcdf(d::InverseGamma, x::Real) = logccdf(d.invd, 1 / x)
logccdf(d::InverseGamma, x::Real) = logcdf(d.invd, 1 / x)

quantile(d::InverseGamma, p::Real) = 1 / cquantile(d.invd, p)
cquantile(d::InverseGamma, p::Real) = 1 / quantile(d.invd, p)
invlogcdf(d::InverseGamma, p::Real) = 1 / invlogccdf(d.invd, p)
invlogccdf(d::InverseGamma, p::Real) = 1 / invlogcdf(d.invd, p)

function mgf(d::InverseGamma{T}, t::Real) where T<:Real
    (a, b) = params(d)
    t == zero(t) ? one(T) : 2(-b*t)^(0.5a) / gamma(a) * besselk(a, sqrt(-4*b*t))
end

function cf(d::InverseGamma{T}, t::Real) where T<:Real
    (a, b) = params(d)
    t == zero(t) ? one(T)+zero(T)*im : 2(-im*b*t)^(0.5a) / gamma(a) * besselk(a, sqrt(-4*im*b*t))
end


#### Evaluation

rand(rng::AbstractRNG, d::InverseGamma) = 1 / rand(rng, d.invd)

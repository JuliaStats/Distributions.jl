"""
    Cosine(μ, σ)

A raised Cosine distribution.

External link:

* [Cosine distribution on wikipedia](http://en.wikipedia.org/wiki/Raised_cosine_distribution)
"""
struct Cosine{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    Cosine{T}(μ::T, σ::T) where {T} = new{T}(µ, σ)
end

function Cosine(μ::T, σ::T; check_args::Bool=true) where {T <: Real}
    @check_args Cosine (σ, σ > zero(σ))
    return Cosine{T}(μ, σ)
end

Cosine(μ::Real, σ::Real; check_args::Bool=true) = Cosine(promote(μ, σ)...; check_args=check_args)
Cosine(μ::Integer, σ::Integer; check_args::Bool=true) = Cosine(float(μ), float(σ); check_args=check_args)
Cosine(μ::Real=0.0) = Cosine(μ, one(µ); check_args=false)

@distr_support Cosine d.μ - d.σ d.μ + d.σ

#### Conversions
function convert(::Type{Cosine{T}}, μ::Real, σ::Real) where T<:Real
    Cosine(T(μ), T(σ))
end
function Base.convert(::Type{Cosine{T}}, d::Cosine) where {T<:Real}
    Cosine{T}(T(d.μ), T(d.σ))
end
Base.convert(::Type{Cosine{T}}, d::Cosine{T}) where {T<:Real} = d

#### Parameters

location(d::Cosine) = d.μ
scale(d::Cosine) = d.σ

params(d::Cosine) = (d.μ, d.σ)
@inline partype(d::Cosine{T}) where {T<:Real} = T


#### Statistics

mean(d::Cosine) = d.μ

median(d::Cosine) = d.μ

mode(d::Cosine) = d.μ

var(d::Cosine{T}) where {T<:Real} = d.σ^2 * (1//3 - 2/T(π)^2)

skewness(d::Cosine{T}) where {T<:Real} = zero(T)

kurtosis(d::Cosine{T}) where {T<:Real} = 6*(90-T(π)^4) / (5*(T(π)^2-6)^2)


#### Evaluation

function pdf(d::Cosine{T}, x::S) where {T<:Real, S<:Real}
    W = promote_type(T, S)
    if insupport(d, x)
        z = (x - d.μ) / d.σ
        return (1 + cospi(z)) / (2d.σ)
    else
        return zero(W)
    end
end

logpdf(d::Cosine, x::Real) = log(pdf(d, x))

function cdf(d::Cosine{T}, x::S) where {T<:Real, S<:Real}
    W = promote_type(T, S)
    if x < d.μ - d.σ
        return zero(W)
    end
    if x > d.μ + d.σ
        return one(W)
    end
    z = (x - d.μ) / d.σ
    (1 + z + sinpi(z) * invπ) / 2
end

function cdf(d::Cosine{T}, x::S) where {T<:Real, S<:Real}
    W = promote_type(T, S)
    if x < d.μ - d.σ
        return one(W)
    end
    if x > d.μ + d.σ
        return zero(W)
    end
    nz = (d.μ - x) / d.σ
    (1 + nz + sinpi(nz) * invπ) / 2
end

quantile(d::Cosine, p::Real) = quantile_bisect(d, p)

import IrrationalConstants: invπ
import LogExpFunctions: logabssinh, log1psq

function mgf(d::Cosine, t::Real)
    σt, μt = d.σ * t, d.μ*t
    z = iszero(σt) ? one(float(σt)) : sinh(σt)/σt
    return exp(μt) * (z / (1 + (invπ * σt)^2))
end

function cgf(d::Cosine, t::Real)
    σt, μt = d.σ * t, d.μ*t
    z = iszero(σt) ? zero(float(σt)) : logabssinh(σt) - log(σt)
    return μt + z - log1psq(invπ * σt)
end

function cf(d::Cosine, t::Real)
    σt, μt = d.σ * t, d.μ*t
    abs(σt) ≈ π && return cis(μt) / 2
    z = iszero(σt) ? one(float(σt)) : sin(σt)/σt
    return π^2 * cis(μt) * z / (π^2 - (σt)^2)
end

#### Affine transformations

Base.:+(d::Cosine, a::Real) = Cosine(d.μ + a, d.σ)
Base.:*(c::Real, d::Cosine) = Cosine(c * d.μ, abs(c) * d.σ)

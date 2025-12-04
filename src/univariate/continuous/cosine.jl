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

function pdf(d::Cosine{T}, x::Real) where T<:Real
    if insupport(d, x)
        z = (x - d.μ) / d.σ
        return (1 + cospi(z)) / (2d.σ)
    else
        return zero(T)
    end
end

logpdf(d::Cosine, x::Real) = log(pdf(d, x))

function cdf(d::Cosine{T}, x::Real) where T<:Real
    if x < d.μ - d.σ
        return zero(T)
    end
    if x > d.μ + d.σ
        return one(T)
    end
    z = (x - d.μ) / d.σ
    (1 + z + sinpi(z) * invπ) / 2
end

function ccdf(d::Cosine{T}, x::Real) where T<:Real
    if x < d.μ - d.σ
        return one(T)
    end
    if x > d.μ + d.σ
        return zero(T)
    end
    nz = (d.μ - x) / d.σ
    (1 + nz + sinpi(nz) * invπ) / 2
end

quantile(d::Cosine, p::Real) = quantile_bisect(d, p)

function mgf(d::Cosine{T}, t::Real) where T<:Real
    μ, σ = params(d)
    t ≈ 0. && return exp(μ*t)
    return T(π^2) * exp(μ*t) * sinh(σ*t) / (σ*t*(T(π^2) + (σ*t)^2))
end

function cgf(d::Cosine{T}, t::Real) where T<:Real
    μ, σ = params(d)
    t ≈ 0. && return μ*t
    return 2log(π) + μ*t + σ*abs(t) + log1p(-exp(-2σ*abs(t))) - log(2) - log(σ*abs(t)) - log(T(π^2) + (σ*t)^2)
end

function cf(d::Cosine{T}, t::Real) where T<:Real
    μ, σ = params(d)
    t ≈ 0. && return one(complex(T))
    σ*abs(t) ≈ π && return cis(μ*t) / 2
    return T(π)^2 * cis(μ*t) * sin(σ*t) / (σ*t*(T(π)^2 - (σ*t)^2))
end

#### Affine transformations

Base.:+(d::Cosine, a::Real) = Cosine(d.μ + a, d.σ)
Base.:*(c::Real, d::Cosine) = Cosine(c * d.μ, abs(c) * d.σ)

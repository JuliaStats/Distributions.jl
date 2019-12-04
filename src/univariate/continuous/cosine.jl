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

function Cosine(μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Cosine, σ > zero(σ))
    return Cosine{T}(μ, σ)
end

Cosine(μ::Real, σ::Real) = Cosine(promote(μ, σ)...)
Cosine(μ::Integer, σ::Integer) = Cosine(float(μ), float(σ))
Cosine(μ::T) where {T <: Real} = Cosine(μ, one(µ))
Cosine() = Cosine(0.0, 1.0, check_args=false)

@distr_support Cosine d.μ - d.σ d.μ + d.σ

#### Conversions
function convert(::Type{Cosine{T}}, μ::Real, σ::Real) where T<:Real
    Cosine(T(μ), T(σ))
end
function convert(::Type{Cosine{T}}, d::Cosine{S}) where {T <: Real, S <: Real}
    Cosine(T(d.μ), T(d.σ), check_args=false)
end

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

function logpdf(d::Cosine{T}, x::Real) where T<:Real
    insupport(d, x) ? log(pdf(d, x)) : -T(Inf)
end

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

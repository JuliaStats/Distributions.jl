# Raised Cosine distribution
#
# Ref: http://en.wikipedia.org/wiki/Raised_cosine_distribution
#

immutable Cosine{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    (::Type{Cosine{T}}){T}(μ::T, σ::T) = (@check_args(Cosine, σ > zero(σ)); new{T}(μ, σ))
end

Cosine{T<:Real}(μ::T, σ::T) = Cosine{T}(μ, σ)
Cosine(μ::Real, σ::Real) = Cosine(promote(μ, σ)...)
Cosine(μ::Integer, σ::Integer) = Cosine(Float64(μ), Float64(σ))
Cosine(μ::Real) = Cosine(μ, 1.0)
Cosine() = Cosine(0.0, 1.0)

@distr_support Cosine d.μ - d.σ d.μ + d.σ

#### Conversions
function convert{T<:Real}(::Type{Cosine{T}}, μ::Real, σ::Real)
    Cosine(T(μ), T(σ))
end
function convert{T <: Real, S <: Real}(::Type{Cosine{T}}, d::Cosine{S})
    Cosine(T(d.μ), T(d.σ))
end

#### Parameters

location(d::Cosine) = d.μ
scale(d::Cosine) = d.σ

params(d::Cosine) = (d.μ, d.σ)
@inline partype{T<:Real}(d::Cosine{T}) = T


#### Statistics

mean(d::Cosine) = d.μ

median(d::Cosine) = d.μ

mode(d::Cosine) = d.μ

var{T<:Real}(d::Cosine{T}) = d.σ^2 * (1//3 - 2/T(π)^2)

skewness{T<:Real}(d::Cosine{T}) = zero(T)

kurtosis{T<:Real}(d::Cosine{T}) = 6*(90-T(π)^4) / (5*(T(π)^2-6)^2)


#### Evaluation

function pdf{T<:Real}(d::Cosine{T}, x::Real)
    if insupport(d, x)
        z = (x - d.μ) / d.σ
        return (1 + cospi(z)) / (2d.σ)
    else
        return zero(T)
    end
end

function logpdf{T<:Real}(d::Cosine{T}, x::Real)
    insupport(d, x) ? log(pdf(d, x)) : -T(Inf)
end

function cdf{T<:Real}(d::Cosine{T}, x::Real)
    if x < d.μ - d.σ 
        return zero(T)
    end
    if x > d.μ + d.σ 
        return one(T)
    end
    z = (x - d.μ) / d.σ
    (1 + z + sinpi(z) * invπ) / 2
end

function ccdf{T<:Real}(d::Cosine, x::Real)
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

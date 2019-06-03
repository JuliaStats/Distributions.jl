"""
    Biweight(μ, σ)
"""
struct Biweight{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    Biweight{T}(μ::T, σ::T) where {T} = (@check_args(Biweight, σ > zero(σ)); new{T}(μ, σ))
end

Biweight(μ::T, σ::T) where {T<:Real} = Biweight{T}(μ, σ)
Biweight(μ::Real, σ::Real) = Biweight(promote(μ, σ)...)
Biweight(μ::Integer, σ::Integer) = Biweight(Float64(μ), Float64(σ))
Biweight(μ::Real) = Biweight(μ, 1.0)
Biweight() = Biweight(0.0, 1.0)

@distr_support Biweight d.μ - d.σ d.μ + d.σ

## Parameters
params(d::Biweight) = (d.μ, d.σ)
@inline partype(d::Biweight{T}) where {T<:Real} = T

## Properties
mean(d::Biweight) = d.μ
median(d::Biweight) = d.μ
mode(d::Biweight) = d.μ

var(d::Biweight) = d.σ^2 / 7
skewness(d::Biweight{T}) where {T<:Real} = zero(T)
kurtosis(d::Biweight{T}) where {T<:Real} = T(1)/21 - 3

## Functions
function pdf(d::Biweight{T}, x::Real) where T<:Real
    u = abs(x - d.μ) / d.σ
    u >= 1 ? zero(T) : (15//16) * (1 - u^2)^2 / d.σ
end

function cdf(d::Biweight{T}, x::Real) where T<:Real
    u = (x - d.μ) / d.σ
    u <= -1 ? zero(T) :
    u >= 1 ? one(T) :
    (u + 1)^3/16 * @horner(u,8,-9,3)
end

function ccdf(d::Biweight{T}, x::Real) where T<:Real
    u = (d.μ - x) / d.σ
    u <= -1 ? one(T) :
    u >= 1 ? zero(T) :
    (u + 1)^3/16 * @horner(u,8,-9,3)
end

@quantile_newton Biweight

function mgf(d::Biweight{T}, t::Real) where T<:Real
    a = d.σ*t
    a2 = a^2
    a == 0 ? one(T) :
    15exp(d.μ * t) * (-3cosh(a) + (a + 3/a) * sinh(a)) / (a2^2)
end

function cf(d::Biweight{T}, t::Real) where T<:Real
    a = d.σ * t
    a2 = a^2
    a == 0 ? one(T)+zero(T)*im :
    -15cis(d.μ * t) * (3cos(a) + (a - 3/a) * sin(a)) / (a2^2)
end

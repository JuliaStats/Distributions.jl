"""
    Biweight(μ, σ)
"""
struct Biweight{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    Biweight{T}(µ::T, σ::T) where {T <: Real} = new{T}(µ, σ)
end

function Biweight(μ::T, σ::T; check_args::Bool=true) where {T<:Real}
    @check_args Biweight (σ, σ > zero(σ))
    return Biweight{T}(μ, σ)
end

Biweight(μ::Real, σ::Real; check_args::Bool=true) = Biweight(promote(μ, σ)...; check_args=check_args)
Biweight(μ::Integer, σ::Integer; check_args::Bool=true) = Biweight(float(μ), float(σ); check_args=check_args)
Biweight(μ::Real=0.0) = Biweight(μ, one(μ); check_args=false)

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
logpdf(d::Biweight, x::Real) = log(pdf(d, x))

function cdf(d::Biweight{T}, x::Real) where T<:Real
    u = (x - d.μ) / d.σ
    u ≤ -1 ? zero(T) :
    u ≥ 1 ? one(T) :
    (u + 1)^3/16 * @horner(u,8,-9,3)
end

function ccdf(d::Biweight{T}, x::Real) where T<:Real
    u = (d.μ - x) / d.σ
    u ≤ -1 ? zero(T) :
    u ≥ 1 ? one(T) :
    (u + 1)^3/16 * @horner(u, 8, -9, 3)
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

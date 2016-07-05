immutable Biweight{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    Biweight(μ::T, σ::T) = (@check_args(Biweight, σ > zero(σ)); new(μ, σ))
end

Biweight{T<:Real}(μ::T, σ::T) = Biweight{T}(μ, σ)
Biweight(μ::Real, σ::Real) = Biweight(promote(μ, σ)...)
Biweight(μ::Integer, σ::Integer) = Biweight(Float64(μ), Float64(σ))
Biweight(μ::Real) = Biweight(μ, 1.0)
Biweight() = Biweight(0.0, 1.0)

@distr_support Biweight d.μ - d.σ d.μ + d.σ

## Parameters
params(d::Biweight) = (d.μ, d.σ)

## Properties
mean(d::Biweight) = d.μ
median(d::Biweight) = d.μ
mode(d::Biweight) = d.μ

var(d::Biweight) = d.σ^2 / 7
skewness{T<:Real}(d::Biweight{T}) = zero(T)
kurtosis{T<:Real}(d::Biweight{T}) = -2.9523809523809526*one(T)  # = 1/21-3

## Functions
function pdf{T<:Real}(d::Biweight{T}, x::Real)
    u = abs(x - d.μ) / d.σ
    u >= 1 ? zero(T) : 0.9375 * (1 - u^2)^2 / d.σ
end

function cdf{T<:Real}(d::Biweight{T}, x::Real)
    u = (x - d.μ) / d.σ
    u <= -1 ? zero(T) :
    u >= 1 ? one(T) :
    0.0625(u + 1)^3 * @horner(u,8,-9,3)
end

function ccdf{T<:Real}(d::Biweight{T}, x::Real)
    u = (d.μ - x) / d.σ
    u <= -1 ? one(T) :
    u >= 1 ? zero(T) :
    0.0625(u + 1)^3 * @horner(u,8,-9,3)
end

@quantile_newton Biweight

function mgf{T<:Real}(d::Biweight{T}, t::Real)
    a = d.σ*t
    a2 = a^2
    a == 0 ? one(T) :
    15exp(d.μ * t) * (-3cosh(a) + (a + 3/a) * sinh(a)) / (a2^2)
end

function cf{T<:Real}(d::Biweight{T}, t::Real)
    a = d.σ * t
    a2 = a^2
    a == 0 ? one(T)+zero(T)*im :
    -15cis(d.μ * t) * (3cos(a) + (a - 3/a) * sin(a)) / (a2^2)
end

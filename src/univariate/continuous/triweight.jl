"""
    Triweight(μ, σ)
"""
struct Triweight{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    Triweight{T}(μ::T, σ::T) where {T} = (@check_args(Triweight, σ > zero(σ)); new{T}(μ, σ))
end

Triweight(μ::T, σ::T) where {T<:Real} = Triweight{T}(μ, σ)
Triweight(μ::Real, σ::Real) = Triweight(promote(μ, σ)...)
Triweight(μ::Integer, σ::Integer) = Triweight(Float64(μ), Float64(σ))
Triweight(μ::Real) = Triweight(μ, 1.0)
Triweight() = Triweight(0.0, 1.0)

@distr_support Triweight d.μ - d.σ d.μ + d.σ

## Conversions

convert(::Type{Triweight{T}}, μ::Real, σ::Real) where {T<:Real} = Triweight(T(μ), T(σ))
convert(::Type{Triweight{T}}, d::Triweight{S}) where {T<:Real, S<:Real} = Triweight(T(d.μ), T(d.σ))

## Parameters

location(d::Triweight) = d.μ
scale(d::Triweight) = d.σ
params(d::Triweight) = (d.μ, d.σ)
@inline partype(d::Triweight{T}) where {T<:Real} = T


## Properties
mean(d::Triweight) = d.μ
median(d::Triweight) = d.μ
mode(d::Triweight) = d.μ

var(d::Triweight) = d.σ^2 / 9
skewness(d::Triweight{T}) where {T<:Real} = zero(T)
kurtosis(d::Triweight{T}) where {T<:Real} = T(1)/33 - 3

## Functions
function pdf(d::Triweight{T}, x::Real) where T<:Real
    u = abs(x - d.μ)/d.σ
    u >= 1 ? zero(T) : 1.09375*(1 - u*u)^3/d.σ
end

function cdf(d::Triweight{T}, x::Real) where T<:Real
    u = (x - d.μ)/d.σ
    u <= -1 ? zero(T) : u >= 1 ? one(T) : 0.03125*(1 + u)^4*@horner(u,16,-29,20,-5)
end

function ccdf(d::Triweight{T}, x::Real) where T<:Real
    u = (d.μ - x)/d.σ
    u <= -1 ? one(T) : u >= 1 ? zero(T) : 0.03125*(1 + u)^4*@horner(u,16,-29,20,-5)
end

@quantile_newton Triweight

function mgf(d::Triweight{T}, t::Float64) where T<:Real
    a = d.σ*t
    a2 = a*a
    a == 0 ? one(T) : 105*exp(d.μ*t)*((15/a2+1)*cosh(a)-(15/a2-6)/a*sinh(a))/(a2*a2)
end

function cf(d::Triweight{T}, t::Float64) where T<:Real
    a = d.σ*t
    a2 = a*a
    a == 0 ? complex(one(T)) : 105*cis(d.μ*t)*((1-15/a2)*cos(a)+(15/a2-6)/a*sin(a))/(a2*a2)
end

"""
    Triweight(μ, σ)
"""
struct Triweight{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    Triweight{T}(µ::T, σ::T) where {T} = new{T}(µ, σ)
end

function Triweight(μ::T, σ::T; check_args::Bool=true) where {T <: Real}
    @check_args Triweight (σ, σ > zero(σ))
    return Triweight{T}(μ, σ)
end

Triweight(μ::Real, σ::Real; check_args::Bool=true) = Triweight(promote(μ, σ)...; check_args=check_args)
Triweight(μ::Integer, σ::Integer; check_args::Bool=true) = Triweight(float(μ), float(σ); check_args=check_args)
Triweight(μ::Real=0.0) = Triweight(μ, one(μ); check_args=false)

@distr_support Triweight d.μ - d.σ d.μ + d.σ

## Conversions

convert(::Type{Triweight{T}}, μ::Real, σ::Real) where {T<:Real} = Triweight(T(μ), T(σ))
Base.convert(::Type{Triweight{T}}, d::Triweight) where {T<:Real} = Triweight{T}(T(d.μ), T(d.σ))
Base.convert(::Type{Triweight{T}}, d::Triweight{T}) where {T<:Real} = d

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
logpdf(d::Triweight, x::Real) = log(pdf(d, x))

function cdf(d::Triweight{T}, x::Real) where T<:Real
    u = (x - d.μ)/d.σ
    u ≤ -1 ? zero(T) : u ≥ 1 ? one(T) : 0.03125*(1 + u)^4*@horner(u,16,-29,20,-5)
end

function ccdf(d::Triweight{T}, x::Real) where T<:Real
    u = (d.μ - x)/d.σ
    u ≤ -1 ? zero(T) : u ≥ 1 ? one(T) : 0.03125*(1 + u)^4*@horner(u,16,-29,20,-5)
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

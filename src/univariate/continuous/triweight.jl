immutable Triweight{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    Triweight(μ::T, σ::T) = (@check_args(Triweight, σ > zero(σ)); new(μ, σ))
end

Triweight{T<:Real}(μ::T, σ::T) = Triweight{T}(μ, σ)
Triweight(μ::Real, σ::Real) = Triweight(promote(μ, σ)...)
Triweight(μ::Integer, σ::Integer) = Triweight(Float64(μ), Float64(σ))
Triweight(μ::Real) = Triweight(μ, 1.0)
Triweight() = Triweight(0.0, 1.0)

@distr_support Triweight d.μ - d.σ d.μ + d.σ

## Conversions

convert{T<:Real}(::Type{Triweight{T}}, μ::Real, σ::Real) = Triweight(T(μ), T(σ))
convert{T <: Real, S <: Real}(::Type{Triweight{T}}, d::Triweight{S}) = Triweight(T(d.μ), T(d.σ))

## Parameters

location(d::Triweight) = d.μ
scale(d::Triweight) = d.σ
params(d::Triweight) = (d.μ, d.σ)


## Properties
mean(d::Triweight) = d.μ
median(d::Triweight) = d.μ
mode(d::Triweight) = d.μ

var(d::Triweight) = d.σ^2 / 9
skewness{T<:Real}(d::Triweight{T}) = zero(T)
kurtosis{T<:Real}(d::Triweight{T}) = T(-2.9696969696969697)  # 1/33-3

## Functions
function pdf{T<:Real}(d::Triweight{T}, x::Real)
    u = abs(x - d.μ)/d.σ
    u >= 1 ? zero(T) : 1.09375*(1 - u*u)^3/d.σ
end

function cdf{T<:Real}(d::Triweight{T}, x::Real)
    u = (x - d.μ)/d.σ
    u <= -1 ? zero(T) : u >= 1 ? one(T) : 0.03125*(1 + u)^4*@horner(u,16,-29,20,-5)
end

function ccdf{T<:Real}(d::Triweight{T}, x::Real)
    u = (d.μ - x)/d.σ
    u <= -1 ? one(T) : u >= 1 ? zero(T) : 0.03125*(1 + u)^4*@horner(u,16,-29,20,-5)
end

@quantile_newton Triweight

function mgf{T<:Real}(d::Triweight{T}, t::Float64)
    a = d.σ*t
    a2 = a*a
    a == 0 ? one(T) : 105*exp(d.μ*t)*((15/a2+1)*cosh(a)-(15/a2-6)/a*sinh(a))/(a2*a2)
end

function cf{T<:Real}(d::Triweight{T}, t::Float64)
    a = d.σ*t
    a2 = a*a
    a == 0 ? complex(one(T)) : 105*cis(d.μ*t)*((1-15/a2)*cos(a)+(15/a2-6)/a*sin(a))/(a2*a2)
end

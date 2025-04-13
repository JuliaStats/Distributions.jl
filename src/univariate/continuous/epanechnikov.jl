"""
    Epanechnikov(μ, σ)
"""
struct Epanechnikov{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    Epanechnikov{T}(µ::T, σ::T) where {T} = new{T}(µ, σ)
end

function Epanechnikov(μ::T, σ::T; check_args::Bool=true) where {T<:Real}
    @check_args Epanechnikov (σ, σ > zero(σ))
    return Epanechnikov{T}(μ, σ)
end

Epanechnikov(μ::Real, σ::Real; check_args::Bool=true) = Epanechnikov(promote(μ, σ)...; check_args=check_args)
Epanechnikov(μ::Integer, σ::Integer; check_args::Bool=true) = Epanechnikov(float(μ), float(σ); check_args=check_args)
Epanechnikov(μ::Real=0.0) = Epanechnikov(μ, one(μ); check_args=false)

@distr_support Epanechnikov d.μ - d.σ d.μ + d.σ

#### Conversions
function convert(::Type{Epanechnikov{T}}, μ::Real, σ::Real) where T<:Real
    Epanechnikov(T(μ), T(σ), check_args=false)
end
function Base.convert(::Type{Epanechnikov{T}}, d::Epanechnikov) where {T<:Real}
    Epanechnikov{T}(T(d.μ), T(d.σ))
end
Base.convert(::Type{Epanechnikov{T}}, d::Epanechnikov{T}) where {T<:Real} = d

## Parameters

location(d::Epanechnikov) = d.μ
scale(d::Epanechnikov) = d.σ
params(d::Epanechnikov) = (d.μ, d.σ)
@inline partype(d::Epanechnikov{T}) where {T<:Real} = T

## Properties
mean(d::Epanechnikov) = d.μ
median(d::Epanechnikov) = d.μ
mode(d::Epanechnikov) = d.μ

var(d::Epanechnikov) = d.σ^2 / 5
skewness(d::Epanechnikov{T}) where {T<:Real} = zero(T)
kurtosis(d::Epanechnikov{T}) where {T<:Real} = -2.914285714285714*one(T)  # 3/35-3

## Functions
function pdf(d::Epanechnikov{T}, x::Real) where T<:Real
    u = abs(x - d.μ) / d.σ
    u >= 1 ? zero(T) : 3 * (1 - u^2) / (4 * d.σ)
end
logpdf(d::Epanechnikov, x::Real) = log(pdf(d, x))

function cdf(d::Epanechnikov{T}, x::Real) where T<:Real
    u = (x - d.μ) / d.σ
    u <= -1 ? zero(T) :
    u >= 1 ? one(T) :
    1//2 + u * (3//4 - u^2/4)
end

function ccdf(d::Epanechnikov{T}, x::Real) where T<:Real
    u = (d.μ - x) / d.σ
    u <= -1 ? zero(T) :
    u >= 1 ? one(T) :
    1//2 + u * (3//4 - u^2/4)
end

@quantile_newton Epanechnikov

function mgf(d::Epanechnikov{T}, t::Real) where T<:Real
    a = d.σ * t
    a == 0 ? one(T) :
    3exp(d.μ * t) * (cosh(a) - sinh(a) / a) / a^2
end

function cf(d::Epanechnikov{T}, t::Real) where T<:Real
    a = d.σ * t
    a == 0 ? one(T)+zero(T)*im :
    -3exp(im * d.μ * t) * (cos(a) - sin(a) / a) / a^2
end

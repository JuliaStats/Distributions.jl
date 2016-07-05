immutable Epanechnikov{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    Epanechnikov(μ::T, σ::T) = (@check_args(Epanechnikov, σ > zero(σ)); new(μ, σ))
end

Epanechnikov{T<:Real}(μ::T, σ::T) = Epanechnikov{T}(μ, σ)
Epanechnikov(μ::Real, σ::Real) = Epanechnikov(promote(μ, σ)...)
Epanechnikov(μ::Integer, σ::Integer) = Epanechnikov(Float64(μ), Float64(σ))
Epanechnikov(μ::Real) = Epanechnikov(μ, 1.0)
Epanechnikov() = Epanechnikov(0.0, 1.0)


@distr_support Epanechnikov d.μ - d.σ d.μ + d.σ

#### Conversions
function convert{T<:Real}(::Type{Epanechnikov{T}}, μ::Real, σ::Real)
    Epanechnikov(T(μ), T(σ))
end
function convert{T <: Real, S <: Real}(::Type{Epanechnikov{T}}, d::Epanechnikov{S})
    Epanechnikov(T(d.μ), T(d.σ))
end

## Parameters

location(d::Epanechnikov) = d.μ
scale(d::Epanechnikov) = d.σ
params(d::Epanechnikov) = (d.μ, d.σ)

## Properties
mean(d::Epanechnikov) = d.μ
median(d::Epanechnikov) = d.μ
mode(d::Epanechnikov) = d.μ

var(d::Epanechnikov) = d.σ^2 / 5
skewness{T<:Real}(d::Epanechnikov{T}) = zero(T)
kurtosis{T<:Real}(d::Epanechnikov{T}) = -2.914285714285714*one(T)  # 3/35-3

## Functions
function pdf{T<:Real}(d::Epanechnikov{T}, x::Real)
    u = abs(x - d.μ) / d.σ
    u >= 1 ? zero(T) : (3/4) * (1 - u^2) / d.σ
end

function cdf{T<:Real}(d::Epanechnikov{T}, x::Real)
    u = (x - d.μ) / d.σ
    u <= -1 ? one(T) :
    u >= 1 ? zero(T) :
    1//2 + u * (3//4 - u^2/4)
end

function ccdf{T<:Real}(d::Epanechnikov{T}, x::Real)
    u = (d.μ - x) / d.σ
    u <= -1 ? one(T) :
    u >= 1 ? zero(T) :
    1//2 + u * (3//4 - u^2/4)
end

@quantile_newton Epanechnikov

function mgf{T<:Real}(d::Epanechnikov{T}, t::Real)
    a = d.σ * t
    a == 0 ? one(T) :
    3exp(d.μ * t) * (cosh(a) - sinh(a) / a) / a^2
end

function cf{T<:Real}(d::Epanechnikov{T}, t::Real)
    a = d.σ * t
    a == 0 ? one(T)+zero(T)*im :
    -3exp(im * d.μ * t) * (cos(a) - sin(a) / a) / a^2
end

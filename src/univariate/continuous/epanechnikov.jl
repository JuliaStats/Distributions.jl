immutable Epanechnikov <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64

    Epanechnikov(μ::Real, σ::Real) = (@check_args(Epanechnikov, σ > zero(σ)); new(μ, σ))
    Epanechnikov(μ::Real) = new(μ, 1.0)
    Epanechnikov() = new(0.0, 1.0)
end

@distr_support Epanechnikov d.μ - d.σ d.μ + d.σ

## Parameters

location(d::Epanechnikov) = d.μ
scale(d::Epanechnikov) = d.σ
params(d::Epanechnikov) = (d.μ, d.σ)

## Properties
mean(d::Epanechnikov) = d.μ
median(d::Epanechnikov) = d.μ
mode(d::Epanechnikov) = d.μ

var(d::Epanechnikov) = d.σ^2 / 5
skewness(d::Epanechnikov) = 0.0
kurtosis(d::Epanechnikov) = -2.914285714285714  # 3/35-3

## Functions
function pdf(d::Epanechnikov, x::Float64)
    u = abs(x - d.μ) / d.σ
    u >= 1 ? 0.0 : 0.75 * (1 - u^2) / d.σ
end

function cdf(d::Epanechnikov, x::Float64)
    u = (x - d.μ) / d.σ
    u <= -1 ? 0.0 :
    u >= 1 ? 1.0 :
    0.5 + u * (0.75 - 0.25 * u^2)
end

function ccdf(d::Epanechnikov, x::Float64)
    u = (d.μ - x) / d.σ
    u <= -1 ? 1.0 :
    u >= 1 ? 0.0 :
    0.5 + u * (0.75 - 0.25 * u^2)
end

@quantile_newton Epanechnikov

function mgf(d::Epanechnikov, t::Float64)
    a = d.σ * t
    a == 0 ? 1.0 :
    3.0 * exp(d.μ * t) * (cosh(a) - sinh(a) / a) / a^2
end

function cf(d::Epanechnikov, t::Float64)
    a = d.σ * t
    a == 0 ? 1.0+0.0im :
    -3.0 * exp(im * d.μ * t) * (cos(a) - sin(a) / a) / a^2
end

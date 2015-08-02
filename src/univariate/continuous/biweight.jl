immutable Biweight <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64

    function Biweight(μ::Real, σ::Real)
        σ > zero(σ) || throw(ArgumentError("scale must be positive"))
        @compat new(Float64(μ), Float64(σ))
    end

    Biweight(μ::Real) = new(μ, 1.0)
    Biweight() = new(0.0, 1.0)
end

@distr_support Biweight d.μ - d.σ d.μ + d.σ

## Parameters
params(d::Biweight) = (d.μ, d.σ)

## Properties
mean(d::Biweight) = d.μ
median(d::Biweight) = d.μ
mode(d::Biweight) = d.μ

var(d::Biweight) = d.σ^2 / 7.0
skewness(d::Biweight) = 0.0
kurtosis(d::Biweight) = -2.9523809523809526  # = 1/21-3

## Functions
function pdf(d::Biweight, x::Float64)
    u = abs(x - d.μ) / d.σ
    u >= 1.0 ? 0.0 : 0.9375 * (1 - u^2)^2 / d.σ
end

function cdf(d::Biweight, x::Float64)
    u = (x - d.μ) / d.σ
    u <= -1.0 ? 0.0 :
    u >= 1.0 ? 1.0 :
    0.0625 * (u + 1.0)^3 * @horner(u,8.0,-9.0,3.0)
end

function ccdf(d::Biweight, x::Float64)
    u = (d.μ - x) / d.σ
    u <= -1.0 ? 1.0 :
    u >= 1.0 ? 0.0 :
    0.0625 * (u + 1.0)^3 * @horner(u,8.0,-9.0,3.0)
end

@quantile_newton Biweight

function mgf(d::Biweight, t::Float64)
    a = d.σ*t
    a2 = a^2
    a == 0 ? 1.0 :
    15.0 * exp(d.μ * t) * (-3.0 * cosh(a) + (a + 3.0/a) * sinh(a)) / (a2^2)
end

function cf(d::Biweight, t::Float64)
    a = d.σ * t
    a2 = a^2
    a == 0 ? 1.0+0.0im :
    -15.0 * cis(d.μ * t) * (3.0 * cos(a) + (a - 3.0/a) * sin(a)) / (a2^2)
end

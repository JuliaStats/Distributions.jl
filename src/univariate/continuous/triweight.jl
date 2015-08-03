immutable Triweight <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64

    Triweight(μ::Real, σ::Real) = (@check_args(Triweight, σ > zero(σ)); new(μ, σ))
    Triweight(μ::Real) = new(μ, 1.0)
    Triweight() = new(0.0, 1.0)
end

@distr_support Triweight d.μ - d.σ d.μ + d.σ


## Parameters

location(d::Triweight) = d.μ
scale(d::Triweight) = d.σ
params(d::Triweight) = (d.μ, d.σ)


## Properties
mean(d::Triweight) = d.μ
median(d::Triweight) = d.μ
mode(d::Triweight) = d.μ

var(d::Triweight) = d.σ^2 / 9.0
skewness(d::Triweight) = 0.0
kurtosis(d::Triweight) = -2.9696969696969697  # 1/33-3

## Functions
function pdf(d::Triweight, x::Real)
    u = abs(x - d.μ)/d.σ
    u >= 1 ? 0.0 : 1.09375*(1-u*u)^3/d.σ
end

function cdf(d::Triweight, x::Real)
    u = (x - d.μ)/d.σ
    u <= -1 ? 0.0 : u >= 1 ? 1.0 : 0.03125*(1+u)^4*@horner(u,16.0,-29.0,20.0,-5.0)
end

function ccdf(d::Triweight, x::Real)
    u = (d.μ - x)/d.σ
    u <= -1 ? 1.0 : u >= 1 ? 0.0 : 0.03125*(1+u)^4*@horner(u,16.0,-29.0,20.0,-5.0)
end

@quantile_newton Triweight

function mgf(d::Triweight, t::Float64)
    a = d.σ*t
    a2 = a*a
    a == 0 ? one(a) : 105.0*exp(d.μ*t)*((15.0/a2+1.0)*cosh(a)-(15.0/a2-6.0)/a*sinh(a))/(a2*a2)
end

function cf(d::Triweight, t::Float64)
    a = d.σ*t
    a2 = a*a
    a == 0 ? complex(one(a)) : 105.0*cis(d.μ*t)*((1.0-15.0/a2)*cos(a)+(15.0/a2-6.0)/a*sin(a))/(a2*a2)
end

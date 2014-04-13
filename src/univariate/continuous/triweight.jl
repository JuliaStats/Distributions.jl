immutable Triweight <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Triweight(l::Real, s::Real)
        s > zero(s) || error("scale must be positive")
        @compat new(Float64(l), Float64(s))
    end
end

Triweight(location::Real) = Triweight(location, 1.0)
Triweight() = Triweight(0.0, 1.0)

@distr_support Triweight d.location-d.scale d.location+d.scale

## Parameters
params(d::Triweight) = (d.location, d.scale)

## Properties
mean(d::Triweight) = d.location
median(d::Triweight) = d.location
mode(d::Triweight) = d.location

var(d::Triweight) = d.scale*d.scale/9.0
skewness(d::Triweight) = 0.0
kurtosis(d::Triweight) = 1/33-3

## Functions
function pdf(d::Triweight, x::Real)
    u = abs(x - d.location)/d.scale
    u >= 1 ? 0.0 : 1.09375*(1-u*u)^3/d.scale
end

function cdf(d::Triweight, x::Real)
    u = (x - d.location)/d.scale
    u <= -1 ? 0.0 : u >= 1 ? 1.0 : 0.03125*(1+u)^4*@horner(u,16.0,-29.0,20.0,-5.0)
end
function ccdf(d::Triweight, x::Real)
    u = (d.location - x)/d.scale
    u <= -1 ? 1.0 : u >= 1 ? 0.0 : 0.03125*(1+u)^4*@horner(u,16.0,-29.0,20.0,-5.0)
end

@quantile_newton Triweight


function mgf(d::Triweight, t::Real)
    a = d.scale*t
    a2 = a*a
    a == 0 ? one(a) : 105.0*exp(d.location*t)*((15.0/a2+1.0)*cosh(a)-(15.0/a2-6.0)/a*sinh(a))/(a2*a2)
end

function cf(d::Triweight, t::Real)
    a = d.scale*t
    a2 = a*a
    a == 0 ? complex(one(a)) : 105.0*cis(d.location*t)*((1.0-15.0/a2)*cos(a)+(15.0/a2-6.0)/a*sin(a))/(a2*a2)
end

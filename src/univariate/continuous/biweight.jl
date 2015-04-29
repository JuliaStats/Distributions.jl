immutable Biweight <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Biweight(l::Real, s::Real)
        s > zero(s) || error("scale must be positive")
        @compat new(Float64(l), Float64(s))
    end
end

Biweight(location::Real) = Biweight(location, 1.0)
Biweight() = Biweight(0.0, 1.0)

@distr_support Biweight d.location-d.scale d.location+d.scale

## Parameters
params(d::Biweight) = (d.location, d.scale)

## Properties
mean(d::Biweight) = d.location
median(d::Biweight) = d.location
mode(d::Biweight) = d.location

var(d::Biweight) = d.scale*d.scale/7.0
skewness(d::Biweight) = 0.0
kurtosis(d::Biweight) = 1/21-3

## Functions
function pdf(d::Biweight, x::Real)
    u = abs(x - d.location)/d.scale
    u >= 1 ? 0.0 : 0.9375*(1-u*u)^2/d.scale
end

function cdf(d::Biweight, x::Real)
    u = (x - d.location)/d.scale
    u <= -1 ? 0.0 : u >= 1 ? 1.0 : 0.0625*(1+u)^3*@horner(u,8.0,-9.0,3.0)
end
function ccdf(d::Biweight, x::Real)
    u = (d.location - x)/d.scale
    u <= -1 ? 1.0 : u >= 1 ? 0.0 : 0.0625*(1+u)^3*@horner(u,8.0,-9.0,3.0)
end

@quantile_newton Biweight

function mgf(d::Biweight, t::Real)
    a = d.scale*t
    a2 = a*a
    a == 0 ? one(a) : 15.0*exp(d.location*t)*(-3.0*cosh(a)+(a+3.0/a)*sinh(a))/(a2*a2)
end
function cf(d::Biweight, t::Real)
    a = d.scale*t
    a2 = a*a
    a == 0 ? complex(one(a)) : -15.0*cis(d.location*t)*(3.0*cos(a)+(a-3.0/a)*sin(a))/(a2*a2)
end

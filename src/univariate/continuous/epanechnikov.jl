immutable Epanechnikov <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Epanechnikov(l::Real, s::Real)
        s > zero(s) || error("scale must be positive")
        @compat new(Float64(l), Float64(s))
    end
end

Epanechnikov(location::Real) = Epanechnikov(location, 1.0)
Epanechnikov() = Epanechnikov(0.0, 1.0)

@distr_support Epanechnikov d.location-d.scale d.location+d.scale

## Parameters
params(d::Epanechnikov) = (d.location, d.scale)

## Properties
mean(d::Epanechnikov) = d.location
median(d::Epanechnikov) = d.location
mode(d::Epanechnikov) = d.location

var(d::Epanechnikov) = d.scale*d.scale/5
skewness(d::Epanechnikov) = 0.0
kurtosis(d::Epanechnikov) = 3/35-3

## Functions
function pdf(d::Epanechnikov, x::Real)   
    u = abs(x - d.location)/d.scale
    u >= 1 ? 0.0 : 0.75*(1-u*u)/d.scale
end
function cdf(d::Epanechnikov, x::Real)
    u = (x - d.location)/d.scale
    u <= -1 ? 0.0 : u >= 1 ? 1.0 : 0.5+u*(0.75-0.25*u*u)
end
function ccdf(d::Epanechnikov, x::Real)
    u = (d.location - x)/d.scale
    u <= -1 ? 1.0 : u >= 1 ? 0.0 : 0.5+u*(0.75-0.25*u*u)
end

@quantile_newton Epanechnikov

function mgf(d::Epanechnikov, t::Real)
    a = d.scale*t
    a == 0 ? one(a) : 3.0*exp(d.location*t)*(cosh(a)-sinh(a)/a)/(a*a)
end

function cf(d::Epanechnikov, t::Real)
    a = d.scale*t
    a == 0 ? complex(one(a)) : -3.0*exp(im*d.location*t)*(cos(a)-sin(a)/a)/(a*a)
end

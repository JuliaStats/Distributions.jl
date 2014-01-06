immutable Logistic <: ContinuousUnivariateDistribution
    location::Real
    scale::Real
    function Logistic(l::Real, s::Real)
    	s > zero(s) || error("scale must be positive")
    	new(float64(l), float64(s))
    end
    Logistic(l::Real) = new(float64(l), 1.0)
    Logistic() = new(0.0, 1.0)
end

@_jl_dist_2p Logistic logis

@continuous_distr_support Logistic -Inf Inf

entropy(d::Logistic) = log(d.scale) + 2.0

kurtosis(d::Logistic) = 1.2

mean(d::Logistic) = d.location

median(d::Logistic) = d.location

mode(d::Logistic) = d.location
modes(d::Logistic) = [d.location]

skewness(d::Logistic) = 0.0

std(d::Logistic) = pi * d.scale / sqrt(3.0)

var(d::Logistic) = (pi * d.scale)^2 / 3.0

function mgf(d::Logistic, t::Real)
    m, b = d.location, d.scale
    exp(t * m) * (pi * b * t) / sin(pi * b * t)
end

function cf(d::Logistic, t::Real)
    m, b = d.location, d.scale
    exp(im * t * m) * (pi * b * t) / sinh(pi * b * t)
end

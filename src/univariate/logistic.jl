immutable Logistic <: ContinuousUnivariateDistribution
    location::Real
    scale::Real
    function Logistic(l::Real, s::Real)
    	if s > 0.0
    		new(float64(l), float64(s))
    	else
    		error("scale must be positive")
    	end
	end
end

Logistic(l::Real) = Logistic(l, 1.0)
Logistic()  = Logistic(0.0, 1.0)

@_jl_dist_2p Logistic logis

insupport(d::Logistic, x::Number) = isreal(x) && isfinite(x)

kurtosis(d::Logistic) = 1.2

mean(d::Logistic) = d.location

median(d::Logistic) = d.location

modes(d::Logistic) = [d.location]

skewness(d::Logistic) = 0.0

std(d::Logistic) = pi * d.scale / sqrt(3.0)

var(d::Logistic) = (pi * d.scale)^2 / 3.0

immutable Weibull <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    function Weibull(sh::Real, sc::Real)
    	if 0.0 < sh && 0.0 < sc
    		new(float64(sh), float64(sc))
    	else
    		error("Both shape and scale must be positive")
    	end
    end
end

Weibull(sh::Real) = Weibull(sh, 1.0)

@_jl_dist_2p Weibull weibull

function cdf(d::Weibull, x::Real)
	if 0.0 < x
		return 1.0 - exp(-((x / d.scale)^d.shape))
	else
		0.0
	end
end

insupport(d::Weibull, x::Number) = real(x) && isfinite(x) && 0.0 <= x

mean(d::Weibull) = d.scale * gamma(1.0 + 1.0 / d.shape)

var(d::Weibull) = d.scale^2 * gamma(1.0 + 2.0 / d.shape) - mean(d)^2

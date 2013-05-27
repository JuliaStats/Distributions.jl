immutable TDist <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom allowed
    function TDist(d::Real)
    	if d > 0.0
    		new(float64(d))
    	else
    		error("df must be positive")
    	end
    end
end

@_jl_dist_1p TDist t

function entropy(d::TDist)
	return ((d.df + 1.0) / 2.0) *
	       (digamma((d.df + 1.0) / 2.0) - digamma((d.df) / 2.0)) +
	       (1.0 / 2.0) * log(d.df) +
	       lbeta(d.df + 1.0, 1.0 / 2.0)
end

insupport(d::TDist, x::Number) = isreal(x) && isfinite(x)

mean(d::TDist) = d.df > 1 ? 0.0 : NaN

median(d::TDist) = 0.0

modes(d::TDist) = [0.0]

function pdf(d::TDist, x::Real)
	return 1.0 / (sqrt(d.df) * beta(0.5, 0.5 * d.df)) *
	       (1.0 + x^2 / d.df)^(-0.5 * (d.df + 1.0))
end

function var(d::TDist)
	if d.df > 2.0
		return d.df / (d.df - 2.0)
	elseif d.df > 1.0
		return Inf
	else
		return NaN
	end
end

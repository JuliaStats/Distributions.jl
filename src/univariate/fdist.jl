immutable FDist <: ContinuousUnivariateDistribution
    ndf::Float64
    ddf::Float64
    function FDist(d1::Real, d2::Real)
    	if d1 > 0 && d2 > 0
    		new(float64(d1), float64(d2))
    	else
    		error("Both numerator and denominator degrees of freedom must be positive")
    	end
    end
end

@_jl_dist_2p FDist f

insupport(d::FDist, x::Number) = isreal(x) && isfinite(x) && 0.0 <= x

mean(d::FDist) = 2.0 < d.ddf ? d.ddf / (d.ddf - 2.0) : NaN

function var(d::FDist)
	if 4.0 < d.ddf
		return 2.0 * d.ddf^2 *
		       (d.ndf + d.ddf - 2.0) /
		       (d.ndf * (d.ddf - 2.0)^2 * (d.ddf - 4.0))
	else
		return NaN
	end
end

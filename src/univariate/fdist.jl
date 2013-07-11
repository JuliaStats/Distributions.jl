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

median(d::FDist) = quantile(d, 0.5)

function modes(d::FDist)
    if d.ndf <= 2
        error("The F distribution has no modes when ndf <= 2")
    end
    return [(d.ndf - 2) / d.ndf * d.ddf / (d.ddf + 2)]
end

function var(d::FDist)
	if 4.0 < d.ddf
		return 2.0 * d.ddf^2 *
		       (d.ndf + d.ddf - 2.0) /
		       (d.ndf * (d.ddf - 2.0)^2 * (d.ddf - 4.0))
	else
		return NaN
	end
end

function skewness(d::FDist)
    if d.ddf <= 6
        error("Skewness not defined when ddf <= 6")
    end
    return (2 * d.ndf + d.ddf - 2) * sqrt(8 * (d.ddf - 4)) /
           ((d.ddf - 6) * sqrt(d.ndf * (d.ndf + d.ddf - 2)))
end

function kurtosis(d::FDist)
    if d.ddf <= 8
        error("Kurtosis not defined when ddf <= 8")
    end
    a = d.ndf * (5 * d.ddf - 22) * (d.ndf + d.ddf - 2) +
        (d.ddf - 4) * (d.ddf - 2)^2
    b = d.ndf * (d.ddf - 6) * (d.ddf - 8) * (d.ndf + d.ddf - 2)
    return 12 * a / b
end

entropy(d::FDist) = (log(d.ddf) -log(d.ndf) 
                     +lgamma(0.5*d.ndf) +lgamma(0.5*d.ddf) -lgamma(0.5*(d.ndf+d.ddf)) 
                     +(1.0-0.5*d.ndf)*digamma(0.5*d.ndf) +(-1.0-0.5*d.ddf)*digamma(0.5*d.ddf)
                     +0.5*(d.ndf+d.ddf)*digamma(0.5*(d.ndf+d.ddf)))

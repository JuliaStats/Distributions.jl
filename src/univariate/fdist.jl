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

insupport(::FDist, x::Number) = zero(x) <= x < Inf
insupport(::Type{FDist}, x::Number) = zero(x) <= x < Inf

mean(d::FDist) = 2.0 < d.ddf ? d.ddf / (d.ddf - 2.0) : NaN

median(d::FDist) = quantile(d, 0.5)

modes(d::FDist) = d.ndf <= 2 ? [0.0] : [(d.ndf - 2) / d.ndf * d.ddf / (d.ddf + 2)]

var(d::FDist) = d.ddf > 4.0 ?  2.0 * d.ddf^2 *
		       (d.ndf + d.ddf - 2.0) /
		       (d.ndf * (d.ddf - 2.0)^2 * (d.ddf - 4.0)) : NaN

skewness(d::FDist) = d.ddf > 6 ?  (2 * d.ndf + d.ddf - 2) * sqrt(8 * (d.ddf - 4)) /
           ((d.ddf - 6) * sqrt(d.ndf * (d.ndf + d.ddf - 2))) : NaN

function kurtosis(d::FDist)
    if d.ddf <= 8
        return NaN
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

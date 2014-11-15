immutable NoncentralF <: ContinuousUnivariateDistribution
    ndf::Float64
    ddf::Float64
    ncp::Float64
    function NoncentralF(n::Real, d::Real, nc::Real)
	n > zero(n) && d > zero(d) && nc >= zero(nc) ||
	    error("ndf and ddf must be > 0 and ncp >= 0")
	new(float64(n), float64(d), float64(nc))
    end
end

@_jl_dist_3p NoncentralF nf

@continuous_distr_support NoncentralF 0.0 Inf

mean(d::NoncentralF) = d.ddf > 2.0 ? d.ddf / (d.ddf - 2.0) * (d.ndf + d.ncp) / d.ndf : NaN

var(d::NoncentralF) = d.ddf > 4.0 ? 2.0 * d.ddf^2 *
		       ((d.ndf+d.ncp)^2 + (d.ddf - 2.0)*(d.ndf + 2.0*d.ncp)) /
		       (d.ndf * (d.ddf - 2.0)^2 * (d.ddf - 4.0)) : NaN

function rand(d::NoncentralF)
    rn = rand(NoncentralChisq(d.ndf,d.ncp)) / d.ndf
    rd = rand(Chisq(d.ddf)) / d.ddf
    rn / rd
end

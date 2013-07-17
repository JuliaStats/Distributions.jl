immutable NoncentralF <: ContinuousUnivariateDistribution
    ndf::Float64
    ddf::Float64
    ncp::Float64
    function NoncentralF(n::Real, d::Real, nc::Real)
		if n > 0.0 && d > 0.0 && nc >= 0.0
			new(float64(n), float64(d), float64(nc))
		else
			error("ndf and ddf must be > 0 and ncp >= 0")
		end
    end
end

@_jl_dist_3p NoncentralF nf

mean(d::NoncentralF) = d.ddf > 2.0 ? d.ddf / (d.ddf - 2.0) * (d.ndf + d.ncp) / d.ndf : NaN

var(d::NoncentralF) = d.ddf > 4.0 ? 2.0 * d.ddf^2 *
		       ((d.ndf+d.ncp)^2 + (d.ddf - 2.0)*(d.ndf + 2.0*d.ncp)) /
		       (d.ndf * (d.ddf - 2.0)^2 * (d.ddf - 4.0)) : NaN

entropy(d::NoncentralF) = NaN


insupport(::NoncentralF, x::Number) = zero(x) <= x < Inf
insupport(::Type{NoncentralF}, x::Number) = zero(x) <= x < Inf

function rand(d::NoncentralF)
    rn = rand(NoncentralChisq(d.ndf,d.ncp)) / d.ndf
    rd = rand(Chisq(d.ddf)) / d.ddf
    rn / rd
end

immutable NoncentralF <: ContinuousUnivariateDistribution
    ndf::Float64
    ddf::Float64
    ncp::Float64
    function NonCentralF(n::Real, d::Real, nc::Real)
		if n > 0.0 && d > 0.0 && nc >= 0.0
			new(float64(n), float64(d), float64(nc))
		else
			error("ndf and ddf must be > 0 and ncp >= 0")
		end
    end
end

@_jl_dist_3p NoncentralF nf

insupport(::NoncentralF, x::Number) = zero(x) <= x < Inf
insupport(::Type{NoncentralF}, x::Number) = zero(x) <= x < Inf

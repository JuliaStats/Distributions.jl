immutable NoncentralT <: ContinuousUnivariateDistribution
    df::Float64
    ncp::Float64
    function NonCentralT(d::Real, nc::Real)
    	if d >= 0.0 && nc >= 0.0
    		new(float64(d), float64(nc))
    	else
    		error("df and ncp must be non-negative")
    	end
    end
end

@_jl_dist_2p NoncentralT nt

insupport(d::NoncentralT, x::Number) = isreal(x) && isfinite(x)

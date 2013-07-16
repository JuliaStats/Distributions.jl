immutable NoncentralChisq <: ContinuousUnivariateDistribution
    df::Float64
    ncp::Float64
    function NoncentralChisq(d::Real, nc::Real)
    	if d >= 0.0 && nc >= 0.0
    		new(float64(d), float64(nc))
    	else
    		error("df and ncp must be non-negative")
    	end
    end
end

@_jl_dist_2p NoncentralChisq nchisq

insupport(::NoncentralChisq, x::Real) = zero(x) < x < Inf
insupport(::Type{NoncentralChisq}, x::Real) = zero(x) < x < Inf

immutable NoncentralT <: ContinuousUnivariateDistribution
    df::Float64
    ncp::Float64
    function NoncentralT(d::Real, nc::Real)
    	d >= zero(d) && nc >= zero(nc) || error("df and ncp must be non-negative")
        new(float64(d), float64(nc))
    end
end

@_jl_dist_2p NoncentralT nt

insupport(::NoncentralT, x::Real) = isfinite(x)
insupport(::Type{NoncentralT}, x::Real) = isfinite(x)

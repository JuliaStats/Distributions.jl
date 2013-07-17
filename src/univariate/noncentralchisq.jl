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

mean(d::NoncentralChisq) = d.df + d.ncp
var(d::NoncentralChisq) = 2.0*(d.df + 2.0*d.ncp)
skewness(d::NoncentralChisq) = 2.0*√2*(d.df + 3.0*d.ncp)/sqrt(d.df + 2.0*d.ncp)^3
kurtosis(d::NoncentralChisq) = 12.0*(d.df + 4.0*d.ncp)/(d.df + 2.0*d.ncp)^2
entropy(d::NoncentralChisq) = NaN

@_jl_dist_2p NoncentralChisq nchisq

insupport(::NoncentralChisq, x::Real) = zero(x) < x < Inf
insupport(::Type{NoncentralChisq}, x::Real) = zero(x) < x < Inf

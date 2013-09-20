immutable NoncentralChisq <: ContinuousUnivariateDistribution
    df::Float64
    ncp::Float64
    function NoncentralChisq(d::Real, nc::Real)
    	d >= zero(d) && nc >= zero(nc) || error("df and ncp must be non-negative")
    	new(float64(d), float64(nc))
    end
end

mean(d::NoncentralChisq) = d.df + d.ncp
var(d::NoncentralChisq) = 2.0*(d.df + 2.0*d.ncp)
skewness(d::NoncentralChisq) = 2.0*√2*(d.df + 3.0*d.ncp)/sqrt(d.df + 2.0*d.ncp)^3
kurtosis(d::NoncentralChisq) = 12.0*(d.df + 4.0*d.ncp)/(d.df + 2.0*d.ncp)^2

@_jl_dist_2p NoncentralChisq nchisq

@continuous_distr_support NoncentralChisq 0.0 Inf


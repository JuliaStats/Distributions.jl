immutable NoncentralT <: ContinuousUnivariateDistribution
    df::Float64
    ncp::Float64
    function NoncentralT(d::Real, nc::Real)
    	d >= zero(d) && nc >= zero(nc) || error("df and ncp must be non-negative")
        new(float64(d), float64(nc))
    end
end

@_jl_dist_2p NoncentralT nt

@continuous_distr_support NoncentralT -Inf Inf

mean(d::NoncentralT) = d.df > 1.0 ? sqrt(0.5*d.df)*d.ncp*gamma(0.5*(d.df-1))/gamma(0.5*d.df) : NaN
var(d::NoncentralT) = d.df > 2.0 ? d.df*(1+d.ncp^2)/(d.df-2.0) - mean(d)^2 : NaN

function rand(d::NoncentralT)
    z = randn()
    v = rand(Chisq(d.df))
    (z+d.ncp)/sqrt(v/d.df)
end

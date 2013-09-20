immutable NoncentralBeta <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    ncp::Float64
    function NoncentralBeta(a::Real, b::Real, nc::Real)
    	a > 0.0 && b > 0.0 && nc >= 0.0 ||
            error("alpha and beta must be > 0 and ncp >= 0")
    	new(float64(a), float64(b), float64(nc))
    end
end

@_jl_dist_3p NoncentralBeta nbeta

@continuous_distr_support NoncentralBeta 0.0 1.0

function rand(d::NoncentralBeta)
    a = rand(NoncentralChisq(2.0 * d.alpha, d.ncp))
    b = rand(Chisq(2.0 * d.beta))
    a / (a+b)
end

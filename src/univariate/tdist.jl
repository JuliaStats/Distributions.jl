immutable TDist <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom allowed
    function TDist(d::Real)
    	d > zero(d) || error("df must be positive")
        new(float64(d))
    end
end

@_jl_dist_1p TDist t

@continuous_distr_support TDist -Inf Inf

function entropy(d::TDist)
    hdf = 0.5*d.df
    hdfph = hdf + 0.5
    hdfph*(digamma(hdfph) - digamma(hdf)) +
        0.5*log(d.df) + lbeta(hdf,0.5)
end

mean(d::TDist) = d.df > 1.0 ? 0.0 : NaN

median(d::TDist) = 0.0

mode(d::TDist) = 0.0

function pdf(d::TDist, x::Real)
    1.0 / (sqrt(d.df) * beta(0.5, 0.5 * d.df)) *
        (1.0 + x^2 / d.df)^(-0.5 * (d.df + 1.0))
end

function var(d::TDist)
    d.df > 2.0 && return d.df / (d.df - 2.0)
    d.df > 1.0 && return Inf
    NaN
end

gradloglik(d::TDist, x::Float64) = -((d.df + 1.0) * x) / (x^2 + d.df)

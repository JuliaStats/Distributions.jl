immutable TDist <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom allowed
    function TDist(d::Real)
    	d > zero(d) ? new(float64(d)) : error("df must be positive")
    end
end

@_jl_dist_1p TDist t

function entropy(d::TDist)
    ((d.df + 1.0) / 2.0) *
        (digamma((d.df + 1.0) / 2.0) - digamma((d.df) / 2.0)) +
        (1.0 / 2.0) * log(d.df) + lbeta(d.df + 1.0, 1.0 / 2.0)
end

insupport(::TDist, x::Real) = isfinite(x)
insupport(::Type{TDist}, x::Real) = isfinite(x)

mean(d::TDist) = d.df > 1 ? 0.0 : NaN

median(d::TDist) = 0.0

modes(d::TDist) = [0.0]

function pdf(d::TDist, x::Real)
    1.0 / (sqrt(d.df) * beta(0.5, 0.5 * d.df)) *
        (1.0 + x^2 / d.df)^(-0.5 * (d.df + 1.0))
end

function var(d::TDist)
    d.df > 2.0 && return d.df / (d.df - 2.0)
    d.df > 1.0 && return Inf
    NaN
end

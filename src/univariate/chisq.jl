immutable Chisq <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom are meaningful
    function Chisq(d::Real)
        d > zero(d) || error("df must be positive")
        new(float64(d))
    end
end

@_jl_dist_1p Chisq chisq

@continuous_distr_support Chisq 0.0 Inf

function entropy(d::Chisq)
    x = d.df / 2.0 + log(2.0) + lgamma(d.df / 2.0)
    x + (1.0 - d.df / 2.0) * digamma(d.df / 2.0)
end

kurtosis(d::Chisq) = 12.0 / d.df

mean(d::Chisq) = d.df

# TODO: Switch to using quantile?
function median(d::Chisq)
    k = d.df
    k * (1.0 - 2.0 / (9.0 * k))^3
end

function mgf(d::Chisq, t::Real)
    k = d.df
    (1.0 - 2.0 * t)^(-k / 2.0)
end

cf(d::Chisq, t::Real) = (1.0 - 2.0 * im * t)^(-d.df / 2.0)

mode(d::Chisq) = d.df > 2.0 ? d.df - 2.0 : 0.0

function gradlogpdf(d::Chisq, x::Real)
  insupport(Chisq, x) ? (d.df / 2.0 - 1) / x - 0.5 : 0.0
end

skewness(d::Chisq) = sqrt(8.0 / d.df)

var(d::Chisq) = 2.0 * d.df

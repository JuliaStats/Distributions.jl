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

function gradloglik(d::Chisq, x::Float64)
  insupport(Chisq, x) ? (d.df / 2.0 - 1) / x - 0.5 : 0.0
end

# rand - the distribution chi^2(df) is 2 * gamma(df / 2)
# for integer n, a chi^2(n) is the sum of n squared standard normals
function rand(d::Chisq)
    d.df == 1 ? randn()^2 : 2.0 * rand(Gamma(d.df / 2.0))
end

function rand!(d::Chisq, A::Array{Float64})
    if d.df == 1
        for i = 1:length(A)
            @inbounds A[i] = randn()^2
        end
    else
        s = GammaSampler(d.df / 2.0)
        for i = 1:length(A)
            @inbounds A[i] = 2.0 * rand(s)
        end
    end
    return A
end

skewness(d::Chisq) = sqrt(8.0 / d.df)

var(d::Chisq) = 2.0 * d.df

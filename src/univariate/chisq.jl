immutable Chisq <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom are meaningful
    function Chisq(d::Real)
        d > zero(d) || error("df must be positive")
        new(float64(d))
    end
end

@_jl_dist_1p Chisq chisq

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
modes(d::Chisq) = [mode(d)]

# rand - the distribution chi^2(df) is 2 * gamma(df / 2)
# for integer n, a chi^2(n) is the sum of n squared standard normals
function rand(d::Chisq)
    d.df == 1 ? randn()^2 : 2.0 * rand(Gamma(d.df / 2.0))
end

function rand!(d::Chisq, A::Array{Float64})
    if d.df == 1
        for i in 1:length(A)
            A[i] = randn()^2
        end
        return A
    end
    if d.df >= 2
        dpar = d.df / 2.0 - 1.0 / 3.0
    else
        error("require degrees of freedom df >= 2")
    end
    cpar = 1.0 / sqrt(9.0 * dpar)
    for i in 1:length(A)
        A[i] = 2.0 * randg2(dpar, cpar)
    end
    A
end

skewness(d::Chisq) = sqrt(8.0 / d.df)

var(d::Chisq) = 2.0 * d.df

### handling support

isupperbounded(d::Union(Chisq, Type{Chisq})) = false
islowerbounded(d::Union(Chisq, Type{Chisq})) = true
isbounded(d::Union(Chisq, Type{Chisq})) = false

hasfinitesupport(d::Union(Chisq, Type{Chisq})) = false
min(d::Union(Chisq, Type{Chisq})) = 0.0
max(d::Union(Chisq, Type{Chisq})) = Inf

insupport(::Union(Chisq, Type{Chisq}), x::Real) = x >= 0.0
immutable Chi <: ContinuousUnivariateDistribution
    df::Float64
    function Chi(df::Real)
        df > zero(df) || error("df must be positive")
        new(float64(df))
    end
end

cdf(d::Chi, x::Real) = cdf(Chisq(d.df),x^2)
ccdf(d::Chi, x::Real) = ccdf(Chisq(d.df),x^2)
logcdf(d::Chi, x::Real) = logcdf(Chisq(d.df),x^2)
logccdf(d::Chi, x::Real) = logccdf(Chisq(d.df),x^2)

quantile(d::Chi,p::Real) = sqrt(quantile(Chisq(d.df),p))
cquantile(d::Chi,p::Real) = sqrt(cquantile(Chisq(d.df),p))
invlogcdf(d::Chi,p::Real) = sqrt(invlogcdf(Chisq(d.df),p))
invlogccdf(d::Chi,p::Real) = sqrt(invlogccdf(Chisq(d.df),p))

mean(d::Chi) = √2 * gamma((d.df + 1.0) / 2.0) / gamma(d.df / 2.0)

function mode(d::Chi)
    d.df >= 1.0 || error("Chi distribution has no mode when df < 1")
    sqrt(d.df - 1)
end
modes(d::Chi) = [mode(d)]

var(d::Chi) = d.df - mean(d)^2

function skewness(d::Chi)
    μ, σ = mean(d), std(d)
    (μ / σ^3) * (1.0 - 2.0 * σ^2)
end

function kurtosis(d::Chi)
    μ, σ, γ = mean(d), std(d), skewness(d)
    (2.0 / σ^2) * (1 - μ * σ * γ - σ^2)
end

function entropy(d::Chi)
    lgamma(k / 2.0) + 0.5 * (k - log(2.0) - (k - 1.0) * digamma(k / 2.0))
end

function pdf(d::Chi, x::Real)
    k = d.df
    (2.0^(1.0 - k / 2.0) * x^(k - 1.0) * exp(-x^2 / 2.0)) / gamma(k / 2.0)
end

function entropy(d::Chi)
    k = d.df
    lgamma(k / 2.0) - log(sqrt(2.0)) -
        ((k - 1.0) / 2.0) * digamma(k / 2.0) + k / 2.0
end

rand(d::Chi) = sqrt(rand(Chisq(d.df)))

### handling support
isupperbounded(d::Union(Chi, Type{Chi})) = false
islowerbounded(d::Union(Chi, Type{Chi})) = true
isbounded(d::Union(Chi, Type{Chi})) = false

hasfinitesupport(d::Union(Chi, Type{Chi})) = false
min(d::Union(Chi, Type{Chi})) = zero(Real)
max(d::Union(Chi, Type{Chi})) = Inf

insupport(::Union(Chi, Type{Chi}), x::Real) = min(Chi) <= x < max(Chi)
immutable Chi <: ContinuousUnivariateDistribution
    chisqd::Chisq

    Chi(df::Real) = new(Chisq(df))
end

@distr_support Chi 0.0 Inf


#### Parameters

dof(d::Chi) = dof(d.chisqd)
params(d::Chi) = (dof(d),)


#### Statistics

mean(d::Chi) = (k = dof(d); sqrt2 * gamma((k + 1.0) / 2.0) / gamma(k / 2.0))

var(d::Chi) = dof(d) - mean(d)^2

function skewness(d::Chi)
    μ, σ = mean(d), std(d)
    (μ / σ^3) * (1.0 - 2.0 * σ^2)
end

function kurtosis(d::Chi)
    μ, σ, γ = mean(d), std(d), skewness(d)
    (2.0 / σ^2) * (1 - μ * σ * γ - σ^2)
end

function entropy(d::Chi)
    k = dof(d)
    lgamma(k / 2.0) - log(sqrt(2.0)) -
        ((k - 1.0) / 2.0) * digamma(k / 2.0) + k / 2.0
end

function mode(d::Chi)
    k = dof(d)
    k >= 1.0 || error("Chi distribution has no mode when df < 1")
    sqrt(k - 1.0)
end


#### Evaluation

pdf(d::Chi, x::Float64) = exp(logpdf(d, x))

function logpdf(d::Chi, x::Float64) 
    k = dof(d)
    (1.0 - 0.5 * k) * logtwo + (k - 1.0) * log(x) - 0.5 * x^2 - lgamma(0.5 * k)
end

gradlogpdf(d::Chi, x::Float64) = x >= 0.0 ? (dof(d) - 1.0) / x - x : 0.0

cdf(d::Chi, x::Float64) = cdf(d.chisqd, x^2)
ccdf(d::Chi, x::Float64) = ccdf(d.chisqd, x^2)
logcdf(d::Chi, x::Float64) = logcdf(d.chisqd, x^2)
logccdf(d::Chi, x::Float64) = logccdf(d.chisqd, x^2)

quantile(d::Chi, p::Float64) = sqrt(quantile(d.chisqd, p))
cquantile(d::Chi, p::Float64) = sqrt(cquantile(d.chisqd, p))
invlogcdf(d::Chi, p::Float64) = sqrt(invlogcdf(d.chisqd, p))
invlogccdf(d::Chi, p::Float64) = sqrt(invlogccdf(d.chisqd, p))


#### Sampling

rand(d::Chi) = sqrt(rand(d.chisqd))

immutable Chi <: ContinuousUnivariateDistribution
    ν::Float64

    Chi(ν::Real) = (@check_args(Chi, ν > zero(ν)); new(ν))
end

@distr_support Chi 0.0 Inf
@distr_boundaries Chi :open :closed

#### Parameters

dof(d::Chi) = d.ν
params(d::Chi) = (d.ν,)


#### Statistics

mean(d::Chi) = (h = d.ν * 0.5; sqrt2 * gamma(h + 0.5) / gamma(h))

var(d::Chi) = d.ν - mean(d)^2
_chi_skewness(μ::Float64, σ::Float64) = (σ2 = σ^2; σ3 = σ2 * σ; (μ / σ3) * (1.0 - 2.0 * σ2))

function skewness(d::Chi)
    μ = mean(d)
    σ = sqrt(d.ν - μ^2)
    _chi_skewness(μ, σ)
end

function kurtosis(d::Chi)
    μ = mean(d)
    σ = sqrt(d.ν - μ^2)
    γ = _chi_skewness(μ, σ)
    (2.0 / σ^2) * (1 - μ * σ * γ - σ^2)
end

entropy(d::Chi) = (ν = d.ν;
    lgamma(ν / 2.0) - 0.5 * logtwo - ((ν - 1.0) / 2.0) * digamma(ν / 2.0) + ν / 2.0)

function mode(d::Chi)
    d.ν >= 1.0 || error("Chi distribution has no mode when ν < 1")
    sqrt(d.ν - 1.0)
end


#### Evaluation

pdf(d::Chi, x::Float64) = exp(logpdf(d, x))

function logpdf(d::Chi, x::Float64)
    if insupport(d, x)
        ν = d.ν
        return (1.0 - 0.5 * ν) * logtwo + (ν - 1.0) * log(x) - 0.5 * x^2 - lgamma(0.5 * ν)
    else
       return -Inf
    end
end

gradlogpdf(d::Chi, x::Float64) = x >= 0.0 ? (d.ν - 1.0) / x - x : 0.0

cdf(d::Chi, x::Float64) = insupport(d,x)?chisqcdf(d.ν, x^2):0.0
ccdf(d::Chi, x::Float64) = insupport(d,x)?chisqccdf(d.ν, x^2):1.0
logcdf(d::Chi, x::Float64) = insupport(d,x)?chisqlogcdf(d.ν, x^2):-Inf
logccdf(d::Chi, x::Float64) = insupport(d,x)?chisqlogccdf(d.ν, x^2):0.0

quantile(d::Chi, p::Float64) = sqrt(chisqinvcdf(d.ν, p))
cquantile(d::Chi, p::Float64) = sqrt(chisqinvccdf(d.ν, p))
invlogcdf(d::Chi, p::Float64) = sqrt(chisqinvlogcdf(d.ν, p))
invlogccdf(d::Chi, p::Float64) = sqrt(chisqinvlogccdf(d.ν, p))


#### Sampling

rand(d::Chi) = sqrt(_chisq_rand(d.ν))

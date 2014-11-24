##############################################################################
#
# REFERENCES: Forbes et al. "Statistical Distributions"
#
##############################################################################

immutable BetaPrime <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    function BetaPrime(a::Float64, b::Float64)
        (a > zero(a) && b > zero(b)) || error("Alpha and beta must be positive")
        new(float64(a), float64(b))
    end
end

BetaPrime() = BetaPrime(1.0, 1.0)

@distr_support BetaPrime 0.0 Inf

function mean(d::BetaPrime)
    d.beta > 1.0 ? d.alpha / (d.beta - 1.0) : NaN
end

mode(d::BetaPrime) = d.alpha > 1.0 ? (d.alpha - 1.0) / (d.beta + 1.0) : 0.0

function pdf(d::BetaPrime, x::Float64)
    α, β = d.alpha, d.beta
    (x^(α - 1.0) * (1.0 + x)^(-(α + β))) / beta(α, β)
end

cdf(d::BetaPrime, x::Float64) = cdf(Beta(d.alpha, d.beta), x / (one(x) + x))
ccdf(d::BetaPrime, x::Float64) = ccdf(Beta(d.alpha, d.beta), x / (one(x) + x))
logcdf(d::BetaPrime, x::Float64) = logcdf(Beta(d.alpha, d.beta), x / (one(x) + x))
logccdf(d::BetaPrime, x::Float64) = logccdf(Beta(d.alpha, d.beta), x / (one(x) + x))

quantile(d::BetaPrime, p::Float64) = (x = quantile(Beta(d.alpha,d.beta),p); x / (1.0-x))
cquantile(d::BetaPrime, p::Float64) = (x = cquantile(Beta(d.alpha,d.beta),p); x / (1.0-x))
invlogcdf(d::BetaPrime, p::Float64) = (x = invlogcdf(Beta(d.alpha,d.beta),p); x / (1.0-x))
invlogccdf(d::BetaPrime, p::Float64) = (x = invlogccdf(Beta(d.alpha,d.beta),p); x / (1.0-x))
    

function rand(d::BetaPrime)
    x = rand(Gamma(d.alpha, 1))
    y = rand(Gamma(d.beta, 1))
    x/y
end

function skewness(d::BetaPrime)
    α, β = d.alpha, d.beta
    β > 3.0 ? (2.0 * (2.0 * α + β - 1))/(β - 3.0) * sqrt((β - 2.0)/(α * (α + β - 1.0))) : NaN
end

function var(d::BetaPrime)
    α, β = d.alpha, d.beta
    β > 2.0 ? (α * (α + β - 1.0)) / ((β - 2.0) * (β - 1.0)^2) : NaN
end

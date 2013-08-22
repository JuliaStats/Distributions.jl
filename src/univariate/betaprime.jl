##############################################################################
#
# REFERENCES: Forbes et al. "Statistical Distributions"
#
##############################################################################

immutable BetaPrime <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    function BetaPrime(a::Real, b::Real)
        (a > zero(a) && b > zero(b)) || error("Alpha and beta must be positive")
        new(float64(a), float64(b))
    end
end

BetaPrime() = BetaPrime(1.0, 1.0)

insupport(::BetaPrime, x::Real) = zero(x) < x
insupport(::Type{BetaPrime}, x::Real) = zero(x) < x

function mean(d::BetaPrime)
    d.beta > 1.0 ? d.alpha / (d.beta - 1.0) : NaN
end

mode(d::BetaPrime) = d.alpha > 1.0 ? (d.alpha - 1.0) / (d.beta + 1.0) : 0.0
modes(d::BetaPrime) = [mode(d)]

function pdf(d::BetaPrime, x::Real)
    α, β = d.alpha, d.beta
    (x^(α - 1.0) * (1.0 + x)^(-(α + β))) / beta(α, β)
end

cdf(d::BetaPrime, q::Real) = cdf(Beta(d.alpha, d.beta), q / (one(q) + q))
ccdf(d::BetaPrime, q::Real) = ccdf(Beta(d.alpha, d.beta), q / (one(q) + q))
logcdf(d::BetaPrime, q::Real) = logcdf(Beta(d.alpha, d.beta), q / (one(q) + q))
logccdf(d::BetaPrime, q::Real) = logccdf(Beta(d.alpha, d.beta), q / (one(q) + q))

quantile(d::BetaPrime, p::Real) = (x = quantile(Beta(d.alpha,d.beta),p); x / (1.0-x))
cquantile(d::BetaPrime, p::Real) = (x = cquantile(Beta(d.alpha,d.beta),p); x / (1.0-x))
invlogcdf(d::BetaPrime, p::Real) = (x = invlogcdf(Beta(d.alpha,d.beta),p); x / (1.0-x))
invlogccdf(d::BetaPrime, p::Real) = (x = invlogccdf(Beta(d.alpha,d.beta),p); x / (1.0-x))
    

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

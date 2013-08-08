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

cdf(d::BetaPrime, q::Real) = inc_beta(q / 1.0 + q, d.alpha, d.beta)

insupport(::BetaPrime, x::Real) = zero(x) < x
insupport(::Type{BetaPrime}, x::Real) = zero(x) < x

function mean(d::BetaPrime)
    d.beta > 1.0 || error("mean not defined when beta <= 1")
    d.alpha / (d.beta - 1.0)
end

mode(d::BetaPrime) = d.alpha > 1.0 ? (d.alpha - 1.0) / (d.beta + 1.0) : 0.0
modes(d::BetaPrime) = [mode(d)]

function pdf(d::BetaPrime, x::Real)
    α, β = d.alpha, d.beta
    (x^(α - 1.0) * (1.0 + x)^(-(α + β))) / beta(α, β)
end

rand(d::BetaPrime) = 1.0 / rand(Beta(d.alpha, d.beta))

function skewness(d::BetaPrime)
    α, β = d.alpha, d.beta
    β > 3.0 || error("skewness not defined when β <= 3")
    (2.0 * (2.0 * α + β - 1))/(β - 3.0) * sqrt((β - 2.0)/(α * (α + β - 1.0)))
end

function var(d::BetaPrime)
    α, β = d.alpha, d.beta
    β > 2.0 || error("var not defined when β <= 2")
    (α * (α + β - 1.0)) / ((β - 2.0) * (β - 1.0)^2)
end

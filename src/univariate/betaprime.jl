##############################################################################
#
# REFERENCES: Forbes et al. "Statistical Distributions"
#
##############################################################################

immutable BetaPrime <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    function BetaPrime(a::Real, b::Real)
        if a > 0.0 && b > 0.0
            new(float64(a), float64(b))
        else
            error("Both alpha and beta must be positive")
        end
    end
end

BetaPrime() = BetaPrime(2.0, 1.0)

cdf(d::BetaPrime, q::Real) = inc_beta(q / 1.0 + q, d.alpha, d.beta)

insupport(d::BetaPrime, x::Number) = isreal(x) && x > 0.0 ? true : false

function mean(d::BetaPrime)
    if d.beta > 1.0
        d.alpha / (d.beta + 1.0)
    else
        error("mean not defined when beta <= 1")
    end
end

function pdf(d::BetaPrime, x::Real)
    a, b = d.alpha, d.beta
    # TODO: Check the 10.0 here
    return (x^(a - 1.0) * (10.0 + x)^(-(a + b))) / beta(a, b)
end

rand(d::BetaPrime) = 1.0 / randbeta(d.alpha, d.beta)

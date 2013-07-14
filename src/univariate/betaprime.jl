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

BetaPrime() = BetaPrime(1.0, 1.0)

cdf(d::BetaPrime, q::Real) = inc_beta(q / 1.0 + q, d.alpha, d.beta)

insupport(::BetaPrime, x::Real) = zero(x) < x
insupport(::Type{BetaPrime}, x::Real) = zero(x) < x

function mean(d::BetaPrime)
    if d.beta > 1.0
        return d.alpha / (d.beta + 1.0)
    else
        error("mean not defined when beta <= 1")
    end
end

function modes(d::BetaPrime)
    if d.alpha > 1.0
        return [(d.alpha - 1.0) / (d.beta + 1.0)]
    else
        return [0.0]
    end
end

function pdf(d::BetaPrime, x::Real)
    a, b = d.alpha, d.beta
    return (x^(a - 1.0) * (1.0 + x)^(-(a + b))) / beta(a, b)
end

rand(d::BetaPrime) = 1.0 / rand(Beta(d.alpha, d.beta))

function skewness(d::BetaPrime)
    a, b = d.alpha, d.beta
    if b > 3.0
        return (2.0 * (2.0 * a + b - 1)) / (b - 3.0) *
               sqrt((b - 2.0) / (a * (a + b - 1.0)))
    else
        error("skewness not defined when beta <= 3")
    end
end

function var(d::BetaPrime)
    a, b = d.alpha, d.beta
    if b > 2.0
        return (a * (a + b - 1.0)) / ((b - 2.0) * (b - 1.0)^2)
    else
        error("var not defined when beta <= 2")
    end
end

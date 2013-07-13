##############################################################################
#
# REFERENCES: Wasserman, "All of Statistics"
#
# NOTES: Fails CDF/Quantile matching test
#
##############################################################################

immutable Bernoulli <: DiscreteUnivariateDistribution
    p0::Float64
    p1::Float64

    function Bernoulli(p::Real)
        if 0.0 <= p <= 1.0
            new(1.0 - p, float(p))
        else
            error("prob must be in [0,1]")
        end
    end
end

Bernoulli() = Bernoulli(0.5)

min(d::Bernoulli) = 0
max(d::Bernoulli) = 1

cdf(d::Bernoulli, q::Real) = q >= 0. ? (q >= 1. ? 1.0 : d.p0) : 0.

function entropy(d::Bernoulli) 
    p0 = d.p0
    p1 = d.p1
    p0 == 0. || p0 == 1. ? 0. : -(p0 * log(p0) + p1 * log(p1))
end

insupport(d::Bernoulli, x::Number) = (x == 0) || (x == 1)

mean(d::Bernoulli) = d.p1

var(d::Bernoulli) = d.p0 * d.p1

skewness(d::Bernoulli) = (d.p0 - d.p1) / sqrt(d.p0 * d.p1)

kurtosis(d::Bernoulli) = 1.0 / (d.p0 * d.p1) - 6.0

median(d::Bernoulli) = d.p1 < 0.5 ? 0.0 : 1.0

mgf(d::Bernoulli, t::Real) = d.p0 + d.p1 * exp(t)

cf(d::Bernoulli, t::Real) = d.p0 + d.p1 * exp(im * t)

function modes(d::Bernoulli)
    d.p1 < 0.5 ? [0] : 
    d.p1 > 0.5 ? [1] : [0, 1]
end

pdf(d::Bernoulli, x::Real) = x == 0 ? d.p0 : x == 1 ? d.p1 : 0.0

quantile(d::Bernoulli, p::Real) = 0.0 < p < 1.0 ? (p <= d.p0 ? 0 : 1) : NaN

rand(d::Bernoulli) = rand() > d.p1 ? 0 : 1

function fit_mle(::Type{Bernoulli}, x::Array)
    for i in 1:length(x)
        if !insupport(Bernoulli(), x[i])
            error("Bernoulli observations must be in {0, 1}")
        end
    end
    return Bernoulli(mean(x))
end


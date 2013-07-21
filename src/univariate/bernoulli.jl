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

insupport(::Bernoulli, x::Real) = (x == 0) || (x == 1)
insupport(::Type{Bernoulli}, x::Real) = (x == 0) || (x == 1)

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

quantile(d::Bernoulli, p::Real) = 0.0 <= p <= 1.0 ? (p <= d.p0 ? 0 : 1) : NaN

rand(d::Bernoulli) = rand() > d.p1 ? 0 : 1


## MLE fitting

immutable BernoulliStats <: SufficientStats
    cnt0::Float64
    cnt1::Float64

    BernoulliStats(c0::Real, c1::Real) = new(float64(c0), float64(c1))
end

fit_mle(::Type{Bernoulli}, ss::BernoulliStats) = Bernoulli(ss.cnt1 / (ss.cnt0 + ss.cnt1))

function suffstats{T<:Integer}(::Type{Bernoulli}, x::Array{T})
    n0 = 0
    n1 = 0
    for xi in x
        if xi == 0
            n0 += 1
        elseif xi == 1
            n1 += 1
        else
            throw(DomainError())
        end
    end
    BernoulliStats(n0, n1)
end

function suffstats{T<:Integer}(::Type{Bernoulli}, x::Array{T}, w::Array{Float64})
    n = length(x)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    n0 = 0.0
    n1 = 0.0
    for i = 1:n
        xi = x[i]
        if xi == 0
            n0 += w[i]
        elseif xi == 1
            n1 += w[i]
        else
            throw(DomainError())
        end
    end
    BernoulliStats(n0, n1)
end



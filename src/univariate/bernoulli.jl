##############################################################################
#
# REFERENCES: Wasserman, "All of Statistics"
#
# NOTES: Fails CDF/Quantile matching test
#
##############################################################################

immutable Bernoulli <: DiscreteUnivariateDistribution
    prob::Float64
    function Bernoulli(p::Real)
        if 0.0 <= p <= 1.0
            new(float64(p))
        else
            error("prob must be in [0,1]")
        end
    end
end

Bernoulli() = Bernoulli(0.5)

cdf(d::Bernoulli, q::Real) = q < 0.0 ? 0.0 : (q >= 1.0 ? 1.0 : 1.0 - d.prob)

entropy(d::Bernoulli) = -xlogx(1.0 - d.prob) - xlogx(d.prob)

insupport(d::Bernoulli, x::Number) = (x == 0) || (x == 1)

kurtosis(d::Bernoulli) = 1.0 / var(d) - 6.0

mean(d::Bernoulli) = d.prob

median(d::Bernoulli) = d.prob < 0.5 ? 0.0 : 1.0

function mgf(d::Bernoulli, t::Real)
    p = d.prob
    return 1.0 - p + p * exp(t)
end

function cf(d::Bernoulli, t::Real)
    p = d.prob
    return 1.0 - p + p * exp(im * t)
end

function modes(d::Bernoulli)
    if d.prob < 0.5
      return [0]
    elseif d.prob == 0.5
      return [0, 1]
    else
      return [1]
    end
end

pdf(d::Bernoulli, x::Real) = x == 0 ? (1.0 - d.prob) : (x == 1 ? d.prob : 0.0)

logpdf(d::Bernoulli, mu::Real, y::Real) = y == 0 ? log(1.0 - mu) : (y == 1 ? log(mu) : -Inf)

quantile(d::Bernoulli, p::Real) = 0.0 < p < 1.0 ? (p <= (1.0 - d.prob) ? 0 : 1) : NaN

rand(d::Bernoulli) = rand() > d.prob ? 0 : 1

skewness(d::Bernoulli) = (1.0 - 2.0 * d.prob) / std(d)

var(d::Bernoulli) = d.prob * (1.0 - d.prob)

function fit(::Type{Bernoulli}, x::Array)
    for i in 1:length(x)
        if !insupport(Bernoulli(), x[i])
            error("Bernoulli observations must be in {0, 1}")
        end
    end
    return Bernoulli(mean(x))
end

# GLM methods
function devresid(d::Bernoulli, y::Real, mu::Real, wt::Real)
    2wt * (xlogxdmu(y, mu) + xlogxdmu(1.0 - y, 1.0 - mu))
end

function devresid(d::Bernoulli, y::Vector{Float64}, mu::Vector{Float64}, wt::Vector{Float64})
    [2wt[i] * (xlogxdmu(y[i], mu[i]) + xlogxdmu(1.0 - y[i], 1.0 - mu[i])) for i in 1:length(y)]
end

mustart(d::Bernoulli,  y::Real, wt::Real) = (wt * y + 0.5) / (wt + 1.0)

var(d::Bernoulli, mu::Real) = max(eps(), mu * (1.0 - mu))

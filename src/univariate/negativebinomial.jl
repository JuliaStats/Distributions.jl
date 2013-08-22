# NegativeBinomial is the distribution of the number of failures
# before the r-th success in a sequence of Bernoulli trials.
# We do not enforce integer size, as the distribution is well defined
# for non-integers, and this can be useful for e.g. overdispersed
# discrete survival times.

immutable NegativeBinomial <: DiscreteUnivariateDistribution
    r::Float64
    prob::Float64

    function NegativeBinomial(r::Real, p::Real)
        zero(p) < p <= one(p) || error("prob must be in (0, 1].")
        zero(r) < r || error("r must be positive.")
        new(float64(r), float64(p))
    end

    NegativeBinomial() = new(1.0, 0.5)
end

@_jl_dist_2p NegativeBinomial nbinom

insupport(::NegativeBinomial, x::Real) = isinteger(x) && zero(x) <= x
insupport(::Type{NegativeBinomial}, x::Real) = isinteger(x) && zero(x) <= x

function mgf(d::NegativeBinomial, t::Real)
    r, p = d.r, d.prob
    return ((1.0 - p) * exp(t))^r / (1.0 - p * exp(t))^r
end

function cf(d::NegativeBinomial, t::Real)
    r, p = d.r, d.prob
    return ((1.0 - p) * exp(im * t))^r / (1.0 - p * exp(im * t))^r
end

function mean(d::NegativeBinomial)
    p = d.prob
    (1.0 - p) * d.r / p
end

function var(d::NegativeBinomial)
    p = d.prob
    (1.0 - p) * d.r / (p * p)
end

function std(d::NegativeBinomial)
    p = d.prob
    sqrt((1.0 - p) * d.r) / p
end

function skewness(d::NegativeBinomial)
    p = d.prob
    (2.0 - p) / sqrt((1.0 - p) * d.r)
end

function kurtosis(d::NegativeBinomial)
    p = d.prob
    6.0 / d.r + (p * p) / ((1.0 - p) * d.r)
end

function mode(d::NegativeBinomial)
    p = d.prob
    ifloor((1.0 - p) * (d.r - 1.) / p)
end

modes(d::NegativeBinomial) = [mode(d)]

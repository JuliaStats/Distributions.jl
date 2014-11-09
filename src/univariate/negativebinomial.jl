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

isupperbounded(::Union(NegativeBinomial, Type{NegativeBinomial})) = false
islowerbounded(::Union(NegativeBinomial, Type{NegativeBinomial})) = true
isbounded(::Union(NegativeBinomial, Type{NegativeBinomial})) = false

minimum(::Union(NegativeBinomial, Type{NegativeBinomial})) = 0
maximum(::Union(NegativeBinomial, Type{NegativeBinomial})) = Inf

insupport(::NegativeBinomial, x::Real) = isinteger(x) && zero(x) <= x
insupport(::Type{NegativeBinomial}, x::Real) = isinteger(x) && zero(x) <= x

function probs(d::NegativeBinomial, rgn::UnitRange)
    r = d.r
    p0 = 1.0 - d.prob
    f, l = rgn[1], rgn[end]
    0 <= f <= l || throw(BoundsError())
    res = Array(Float64, l - f + 1)
    res[1] = v = pdf(d, f)
    b = f - 1
    for x = f+1:l
        c = (x + r - 1) * p0 / x
        res[x-b] = (v *= c)
    end
    return res
end

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


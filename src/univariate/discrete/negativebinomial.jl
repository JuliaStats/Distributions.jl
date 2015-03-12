# NegativeBinomial is the distribution of the number of failures
# before the r-th success in a sequence of Bernoulli trials.
# We do not enforce integer size, as the distribution is well defined
# for non-integers, and this can be useful for e.g. overdispersed
# discrete survival times.

immutable NegativeBinomial <: DiscreteUnivariateDistribution
    r::Int
    p::Float64

    function NegativeBinomial(r::Int, p::Float64)
        r > 0 || error("r must be positive.")
        0.0 < p <= 1.0 || error("prob must be in (0, 1].")
        new(r, p)
    end

    @compat NegativeBinomial(r::Real, p::Real) = NegativeBinomial(round(Int, r), Float64(p))
    NegativeBinomial(r::Real) = NegativeBinomial(r, 0.5)
    NegativeBinomial() = new(1.0, 0.5)
end

@_jl_dist_2p NegativeBinomial nbinom

@distr_support NegativeBinomial 0 Inf

### Parameters

params(d::NegativeBinomial) = (d.r, d.p)

succprob(d::NegativeBinomial) = d.p
failprob(d::NegativeBinomial) = 1.0 - d.p


### Statistics

mean(d::NegativeBinomial) = (p = succprob(d); (1.0 - p) * d.r / p)

var(d::NegativeBinomial) = (p = succprob(d); (1.0 - p) * d.r / (p * p))

std(d::NegativeBinomial) = (p = succprob(d); sqrt((1.0 - p) * d.r) / p)

skewness(d::NegativeBinomial) = (p = succprob(d); (2.0 - p) / sqrt((1.0 - p) * d.r))

kurtosis(d::NegativeBinomial) = (p = succprob(d); 6.0 / d.r + (p * p) / ((1.0 - p) * d.r))

mode(d::NegativeBinomial) = (p = succprob(d); floor(Int,(1.0 - p) * (d.r - 1.) / p))


### Evaluation

immutable RecursiveNegBinomProbEvaluator <: RecursiveProbabilityEvaluator
    r::Float64
    p0::Float64
end

RecursiveNegBinomProbEvaluator(d::NegativeBinomial) = RecursiveNegBinomProbEvaluator(d.r, failprob(d))
nextpdf(s::RecursiveNegBinomProbEvaluator, p::Float64, x::Integer) = ((x + s.r - 1) / x) * s.p0 * p
_pdf!(r::AbstractArray, d::NegativeBinomial, rgn::UnitRange) = _pdf!(r, d, rgn, RecursiveNegBinomProbEvaluator(d))

function mgf(d::NegativeBinomial, t::Real)
    r, p = params(d)
    return ((1.0 - p) * exp(t))^r / (1.0 - p * exp(t))^r
end

function cf(d::NegativeBinomial, t::Real)
    r, p = params(d)
    return (((1.0 - p) * cis(t)) / (1.0 - p * cis(t)))^r
end



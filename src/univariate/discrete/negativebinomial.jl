# NegativeBinomial is the distribution of the number of failures
# before the r-th success in a sequence of Bernoulli trials.
# We do not enforce integer size, as the distribution is well defined
# for non-integers, and this can be useful for e.g. overdispersed
# discrete survival times.

immutable NegativeBinomial <: DiscreteUnivariateDistribution
    r::Int
    p::Float64

    function NegativeBinomial(r::Real, p::Real)
        @check_args(NegativeBinomial, r > zero(r))
        @check_args(NegativeBinomial, zero(p) < p <= one(p))
        new(r, p)
    end
    NegativeBinomial(r::Real) = NegativeBinomial(r, 0.5)
    NegativeBinomial() = new(1, 0.5)
end

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


### Evaluation & Sampling

@_delegate_statsfuns NegativeBinomial nbinom r p

rand(d::NegativeBinomial) = convert(Int, StatsFuns.Rmath.nbinomrand(d.r, d.p))

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

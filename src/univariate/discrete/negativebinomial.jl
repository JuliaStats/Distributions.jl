doc"""
    NegativeBinomial(r,p)

A *Negative binomial distribution* describes the number of failures before the `r`th success in a sequence of independent Bernoulli trials. It is parameterized by `r`, the number of successes, and `p`, the probability of success in an individual trial.

$P(X = k) = {k + r - 1 \choose k} p^r (1 - p)^k, \quad \text{for } k = 0,1,2,\ldots.$

The distribution remains well-defined for any positive `r`, in which case

$P(X = k) = \frac{\Gamma(k+r)}{k! \Gamma(r)} p^r (1 - p)^k, \quad \text{for } k = 0,1,2,\ldots.$


```julia
NegativeBinomial()        # Negative binomial distribution with r = 1 and p = 0.5
NegativeBinomial(r, p)    # Negative binomial distribution with r successes and success rate p

params(d)       # Get the parameters, i.e. (r, p)
succprob(d)     # Get the success rate, i.e. p
failprob(d)     # Get the failure rate, i.e. 1 - p
```

External links:

* [Negative binomial distribution on Wikipedia](http://en.wikipedia.org/wiki/Negative_binomial_distribution)

"""

immutable NegativeBinomial{T<:Real} <: DiscreteUnivariateDistribution
    r::T
    p::T

    function NegativeBinomial(r::T, p::T)
        @check_args(NegativeBinomial, r > zero(r))
        @check_args(NegativeBinomial, zero(p) < p <= one(p))
        new(r, p)
    end

end

NegativeBinomial{T<:Real}(r::T, p::T) = NegativeBinomial{T}(r, p)
NegativeBinomial(r::Real, p::Real) = NegativeBinomial(promote(r, p)...)
NegativeBinomial(r::Real) = NegativeBinomial(r, 0.5)
NegativeBinomial() = NegativeBinomial(1.0, 0.5)


@distr_support NegativeBinomial 0 Inf

#### Conversions

function convert{T<:Real}(::Type{NegativeBinomial{T}}, r::Real, p::Real)
    NegativeBinomial(T(r), T(p))
end
function convert{T <: Real, S <: Real}(::Type{NegativeBinomial{T}}, d::NegativeBinomial{S})
    NegativeBinomial(T(d.r), T(d.p))
end

#### Parameters

params(d::NegativeBinomial) = (d.r, d.p)

succprob(d::NegativeBinomial) = d.p
failprob(d::NegativeBinomial) = 1 - d.p


#### Statistics

mean(d::NegativeBinomial) = (p = succprob(d); (1 - p) * d.r / p)

var(d::NegativeBinomial) = (p = succprob(d); (1 - p) * d.r / (p * p))

std(d::NegativeBinomial) = (p = succprob(d); sqrt((1 - p) * d.r) / p)

skewness(d::NegativeBinomial) = (p = succprob(d); (2 - p) / sqrt((1 - p) * d.r))

kurtosis(d::NegativeBinomial) = (p = succprob(d); 6 / d.r + (p * p) / ((1 - p) * d.r))

mode(d::NegativeBinomial) = (p = succprob(d); floor(Int,(1 - p) * (d.r - 1) / p))


#### Evaluation & Sampling

@_delegate_statsfuns NegativeBinomial nbinom r p

rand(d::NegativeBinomial) = convert(Int, StatsFuns.RFunctions.nbinomrand(d.r, d.p))

immutable RecursiveNegBinomProbEvaluator <: RecursiveProbabilityEvaluator
    r::Float64
    p0::Float64
end

RecursiveNegBinomProbEvaluator(d::NegativeBinomial) = RecursiveNegBinomProbEvaluator(d.r, failprob(d))
nextpdf(s::RecursiveNegBinomProbEvaluator, p::Float64, x::Integer) = ((x + s.r - 1) / x) * s.p0 * p
_pdf!(r::AbstractArray, d::NegativeBinomial, rgn::UnitRange) = _pdf!(r, d, rgn, RecursiveNegBinomProbEvaluator(d))

function mgf(d::NegativeBinomial, t::Real)
    r, p = params(d)
    return ((1 - p) * exp(t))^r / (1 - p * exp(t))^r
end

function cf(d::NegativeBinomial, t::Real)
    r, p = params(d)
    return (((1 - p) * cis(t)) / (1 - p * cis(t)))^r
end

"""
    NegativeBinomial(r,p)

A *Negative binomial distribution* describes the number of failures before the `r`th success in a sequence of independent Bernoulli trials. It is parameterized by `r`, the number of successes, and `p`, the probability of success in an individual trial.

```math
P(X = k) = {k + r - 1 \\choose k} p^r (1 - p)^k, \\quad \\text{for } k = 0,1,2,\\ldots.
```

The distribution remains well-defined for any positive `r`, in which case

```math
P(X = k) = \\frac{\\Gamma(k+r)}{k! \\Gamma(r)} p^r (1 - p)^k, \\quad \\text{for } k = 0,1,2,\\ldots.
```

```julia
NegativeBinomial()        # Negative binomial distribution with r = 1 and p = 0.5
NegativeBinomial(r, p)    # Negative binomial distribution with r successes and success rate p

params(d)       # Get the parameters, i.e. (r, p)
succprob(d)     # Get the success rate, i.e. p
failprob(d)     # Get the failure rate, i.e. 1 - p
```

External links:

* [Negative binomial distribution on Wolfram](https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html)
Note: The definition of the negative binomial distribution in Wolfram is different from the [Wikipedia definition](http://en.wikipedia.org/wiki/Negative_binomial_distribution). In Wikipedia, `r` is the number of failures and `k` is the number of successes.

"""
struct NegativeBinomial{T<:Real} <: DiscreteUnivariateDistribution
    r::T
    p::T

    function NegativeBinomial{T}(r::T, p::T) where {T <: Real}
        return new{T}(r, p)
    end
end

function NegativeBinomial(r::T, p::T; check_args=true) where {T <: Real}
    if check_args
        @check_args(NegativeBinomial, r > zero(r))
        @check_args(NegativeBinomial, zero(p) < p <= one(p))
    end
    return NegativeBinomial{T}(r, p)
end

NegativeBinomial(r::Real, p::Real) = NegativeBinomial(promote(r, p)...)
NegativeBinomial(r::Integer, p::Integer) = NegativeBinomial(float(r), float(p))
NegativeBinomial(r::Real) = NegativeBinomial(r, 0.5)
NegativeBinomial() = NegativeBinomial(1.0, 0.5, check_args=false)

@distr_support NegativeBinomial 0 Inf

#### Conversions

function convert(::Type{NegativeBinomial{T}}, r::Real, p::Real) where {T<:Real}
    return NegativeBinomial(T(r), T(p))
end
function convert(::Type{NegativeBinomial{T}}, d::NegativeBinomial{S}) where {T <: Real, S <: Real}
    return NegativeBinomial(T(d.r), T(d.p), check_args=false)
end

#### Parameters

params(d::NegativeBinomial) = (d.r, d.p)
partype(::NegativeBinomial{T}) where {T} = T

succprob(d::NegativeBinomial) = d.p
failprob(d::NegativeBinomial{T}) where {T} = one(T) - d.p


#### Statistics

mean(d::NegativeBinomial{T}) where {T} = (p = succprob(d); (one(T) - p) * d.r / p)

var(d::NegativeBinomial{T}) where {T}  = (p = succprob(d); (one(T) - p) * d.r / (p * p))

std(d::NegativeBinomial{T}) where {T}  = (p = succprob(d); sqrt((one(T) - p) * d.r) / p)

skewness(d::NegativeBinomial{T}) where {T} = (p = succprob(d); (T(2) - p) / sqrt((one(T) - p) * d.r))

kurtosis(d::NegativeBinomial{T}) where {T} = (p = succprob(d); T(6) / d.r + (p * p) / ((one(T) - p) * d.r))

mode(d::NegativeBinomial{T}) where {T} = (p = succprob(d); floor(Int,(one(T) - p) * (d.r - one(T)) / p))


#### Evaluation & Sampling

# Implement native pdf and logpdf since it's relatively straight forward and allows for ForwardDiff
function logpdf(d::NegativeBinomial, k::Int)
    r = d.r*log(d.p) + k*log1p(-d.p)
    if isone(d.p) && iszero(k)
        return zero(r)
    elseif k < 0
        return oftype(r, -Inf)
    else
        return r - log(k + d.r) - Distributions.logbeta(d.r, k + 1)
    end
end
pdf(d::NegativeBinomial, k::Int) = exp(logpdf(d, k))

# cdf and quantile functions are more involved so we still rely on Rmath
cdf(       d::NegativeBinomial,  x::Int)  =              nbinomcdf(       d.r, d.p, x)
ccdf(      d::NegativeBinomial,  x::Int)  =              nbinomccdf(      d.r, d.p, x)
logcdf(    d::NegativeBinomial,  x::Int)  =              nbinomlogcdf(    d.r, d.p, x)
logccdf(   d::NegativeBinomial,  x::Int)  =              nbinomlogccdf(   d.r, d.p, x)
quantile(  d::NegativeBinomial,  q::Real) = convert(Int, nbinominvcdf(    d.r, d.p, q))
cquantile( d::NegativeBinomial,  q::Real) = convert(Int, nbinominvccdf(   d.r, d.p, q))
invlogcdf( d::NegativeBinomial, lq::Real) = convert(Int, nbinominvlogcdf( d.r, d.p, lq))
invlogccdf(d::NegativeBinomial, lq::Real) = convert(Int, nbinominvlogccdf(d.r, d.p, lq))

## sampling
# TODO: remove RFunctions dependency once Poisson has its removed
@rand_rdist(NegativeBinomial)
rand(d::NegativeBinomial) =
    convert(Int, StatsFuns.RFunctions.nbinomrand(d.r, d.p))

function rand(rng::AbstractRNG, d::NegativeBinomial)
    lambda = rand(rng, Gamma(d.r, (1-d.p)/d.p))
    return rand(rng, Poisson(lambda))
end

struct RecursiveNegBinomProbEvaluator <: RecursiveProbabilityEvaluator
    r::Float64
    p0::Float64
end

RecursiveNegBinomProbEvaluator(d::NegativeBinomial) = RecursiveNegBinomProbEvaluator(d.r, failprob(d))
nextpdf(s::RecursiveNegBinomProbEvaluator, p::Float64, x::Integer) = ((x + s.r - 1) / x) * s.p0 * p

Base.broadcast!(::typeof(pdf), r::AbstractArray, d::NegativeBinomial, rgn::UnitRange) =
    _pdf!(r, d, rgn, RecursiveNegBinomProbEvaluator(d))
function Base.broadcast(::typeof(pdf), d::NegativeBinomial, X::UnitRange)
    r = similar(Array{promote_type(partype(d), eltype(X))}, axes(X))
    r .= pdf.(Ref(d),X)
end

function mgf(d::NegativeBinomial, t::Real)
    r, p = params(d)
    return ((1 - p) * exp(t))^r / (1 - p * exp(t))^r
end

function cf(d::NegativeBinomial, t::Real)
    r, p = params(d)
    return (((1 - p) * cis(t)) / (1 - p * cis(t)))^r
end

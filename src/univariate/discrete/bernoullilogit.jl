"""
    BernoulliLogit(logitp=0.0)

A *Bernoulli distribution* that is parameterized by the logit `logitp = logit(p) = log(p/(1-p))` of its success rate `p`.

```math
P(X = k) = \\begin{cases}
\\operatorname{logistic}(-logitp) = \\frac{1}{1 + \\exp{(logitp)}} & \\quad \\text{for } k = 0, \\\\
\\operatorname{logistic}(logitp) = \\frac{1}{1 + \\exp{(-logitp)}} & \\quad \\text{for } k = 1.
\\end{cases}
```

External links:

* [Bernoulli distribution on Wikipedia](http://en.wikipedia.org/wiki/Bernoulli_distribution)

See also [`Bernoulli`](@ref)
"""
struct BernoulliLogit{T<:Real} <: DiscreteUnivariateDistribution
    logitp::T
end

BernoulliLogit() = BernoulliLogit(0.0)

@distr_support BernoulliLogit false true

Base.eltype(::Type{<:BernoulliLogit}) = Bool

#### Conversions
Base.convert(::Type{BernoulliLogit{T}}, d::BernoulliLogit) where {T<:Real} = BernoulliLogit{T}(T(d.logitp))
Base.convert(::Type{BernoulliLogit{T}}, d::BernoulliLogit{T}) where {T<:Real} = d

#### Parameters

succprob(d::BernoulliLogit) = logistic(d.logitp)
failprob(d::BernoulliLogit) = logistic(-d.logitp)
logsuccprob(d::BernoulliLogit) = -log1pexp(-d.logitp)
logfailprob(d::BernoulliLogit) = -log1pexp(d.logitp)

params(d::BernoulliLogit) = (d.logitp,)
partype(::BernoulliLogit{T}) where {T} = T

#### Properties

mean(d::BernoulliLogit) = succprob(d)
var(d::BernoulliLogit) =  succprob(d) * failprob(d)
function skewness(d::BernoulliLogit)
    p0 = failprob(d)
    p1 = succprob(d)
    return (p0 - p1) / sqrt(p0 * p1)
end
kurtosis(d::BernoulliLogit) = 1 / var(d) - 6

mode(d::BernoulliLogit) = d.logitp > 0 ? 1 : 0

function modes(d::BernoulliLogit)
    logitp = d.logitp
    z = zero(logitp)
    logitp < z ? [false] : (logitp > z ? [true] : [false, true])
end

median(d::BernoulliLogit) = d.logitp > 0

function entropy(d::BernoulliLogit)
    logitp = d.logitp
    (logitp == -Inf || logitp == Inf) ? float(zero(logitp)) : (logitp > 0 ? -(succprob(d) * logitp + logfailprob(d)) : -(logsuccprob(d) - failprob(d) * logitp))
end

#### Evaluation

pdf(d::BernoulliLogit, x::Bool) = x ? succprob(d) : failprob(d)
pdf(d::BernoulliLogit, x::Real) = x == 0 ? failprob(d) : (x == 1 ? succprob(d) : zero(float(d.logitp)))

logpdf(d::BernoulliLogit, x::Bool) = x ? logsuccprob(d) : logfailprob(d)
logpdf(d::BernoulliLogit, x::Real) = x == 0 ? logfailprob(d) : (x == 1 ? logsuccprob(d) : oftype(float(d.logitp), -Inf))

cdf(d::BernoulliLogit, x::Bool) = x ? one(float(d.logitp)) : failprob(d)
cdf(d::BernoulliLogit, x::Int) = x < 0 ? zero(float(d.logitp)) : (x < 1 ? failprob(d) : one(float(d.logitp)))

logcdf(d::BernoulliLogit, x::Bool) = x ? zero(float(d.logitp)) : logfailprob(d)
logcdf(d::BernoulliLogit, x::Int) = x < 0 ? oftype(float(d.logitp), -Inf) : (x < 1 ? logfailprob(d) : zero(float(d.logitp)))

ccdf(d::BernoulliLogit, x::Bool) = x ? zero(float(d.logitp)) : succprob(d)
ccdf(d::BernoulliLogit, x::Int) = x < 0 ? one(float(d.logitp)) : (x < 1 ? succprob(d) : zero(float(d.logitp)))

logccdf(d::BernoulliLogit, x::Bool) = x ? oftype(float(d.logitp), -Inf) : logsuccprob(d)
logccdf(d::BernoulliLogit, x::Int) = x < 0 ? zero(float(d.logitp)) : (x < 1 ? logsuccprob(d) : oftype(float(d.logitp), -Inf))

function quantile(d::BernoulliLogit, p::Real)
    T = float(partype(d))
    0 <= p <= 1 ? (p <= failprob(d) ? zero(T) : one(T)) : T(NaN)
end
function cquantile(d::BernoulliLogit, p::Real)
    T = float(partype(d))
    0 <= p <= 1 ? (p >= succprob(d) ? zero(T) : one(T)) : T(NaN)
end

mgf(d::BernoulliLogit, t::Real) = failprob(d) + exp(t + logsuccprob(d))
function cgf(d::BernoulliLogit, t)
    # log(1-p+p*exp(t)) = logaddexp(log(1-p), t + log(p))
    logaddexp(logfailprob(d), t + logsuccprob(d))
end
cf(d::BernoulliLogit, t::Real) = failprob(d) + succprob(d) * cis(t)


#### Sampling

rand(rng::AbstractRNG, d::BernoulliLogit) = logit(rand(rng)) <= d.logitp

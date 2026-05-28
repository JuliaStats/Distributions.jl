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
"""
struct NegativeBinomial{T<:Real} <: DiscreteUnivariateDistribution
    r::T
    p::T

    function NegativeBinomial{T}(r::T, p::T) where {T <: Real}
        return new{T}(r, p)
    end
end

function NegativeBinomial(r::T, p::T; check_args::Bool=true) where {T <: Real}
    @check_args NegativeBinomial (r, r > zero(r)) (p, zero(p) < p <= one(p))
    return NegativeBinomial{T}(r, p)
end

NegativeBinomial(r::Real, p::Real; check_args::Bool=true) = NegativeBinomial(promote(r, p)...; check_args=check_args)
NegativeBinomial(r::Integer, p::Integer; check_args::Bool=true) = NegativeBinomial(float(r), float(p); check_args=check_args)
NegativeBinomial(r::Real; check_args::Bool=true) = NegativeBinomial(r, 0.5; check_args=check_args)
NegativeBinomial() = NegativeBinomial{Float64}(1.0, 0.5)

@distr_support NegativeBinomial 0 Inf

#### Conversions

function convert(::Type{NegativeBinomial{T}}, r::Real, p::Real) where {T<:Real}
    return NegativeBinomial(T(r), T(p))
end
function Base.convert(::Type{NegativeBinomial{T}}, d::NegativeBinomial) where {T<:Real}
    return NegativeBinomial{T}(T(d.r), T(d.p))
end
Base.convert(::Type{NegativeBinomial{T}}, d::NegativeBinomial{T}) where {T<:Real} = d

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

function kldivergence(p::NegativeBinomial, q::NegativeBinomial; kwargs...)
    if p.r == q.r
        return p.r * kldivergence(Geometric(succprob(p)), Geometric(succprob(q)))
    else
        # There does not appear to be an analytical formula for
        # this case. Hence we fall back to the numerical approximation.
        return invoke(kldivergence, Tuple{UnivariateDistribution{Discrete},UnivariateDistribution{Discrete}}, p, q; kwargs...)
    end
end


#### Evaluation & Sampling

# StatsFuns uses Julia implementations for all but (c)quantile and invlog(c)cdf
@_delegate_statsfuns NegativeBinomial nbinom r p

## sampling
function rand(rng::AbstractRNG, d::NegativeBinomial)
    if isone(d.p)
        return 0
    else
        return rand(rng, Poisson(rand(rng, Gamma(d.r, (1 - d.p)/d.p))))
    end
end

function laplace_transform(d::NegativeBinomial, t)
    r, p = params(d)
    return laplace_transform(Geometric(p, check_args=false), t)^r
end

mgf(d::NegativeBinomial, t::Real) = laplace_transform(d, -t)
function cgf(d::NegativeBinomial, t)
    r, p = params(d)
    r * cgf(Geometric{typeof(p)}(p), t)
end
cf(d::NegativeBinomial, t::Real) = laplace_transform(d, -t*im)

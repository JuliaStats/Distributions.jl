"""
    Binomial(n,p)

A *Binomial distribution* characterizes the number of successes in a sequence of independent trials. It has two parameters: `n`, the number of trials, and `p`, the probability of success in an individual trial, with the distribution:

```math
P(X = k) = {n \\choose k}p^k(1-p)^{n-k},  \\quad \\text{ for } k = 0,1,2, \\ldots, n.
```

```julia
Binomial()      # Binomial distribution with n = 1 and p = 0.5
Binomial(n)     # Binomial distribution for n trials with success rate p = 0.5
Binomial(n, p)  # Binomial distribution for n trials with success rate p

params(d)       # Get the parameters, i.e. (n, p)
ntrials(d)      # Get the number of trials, i.e. n
succprob(d)     # Get the success rate, i.e. p
failprob(d)     # Get the failure rate, i.e. 1 - p
```

External links:

* [Binomial distribution on Wikipedia](http://en.wikipedia.org/wiki/Binomial_distribution)
"""
struct Binomial{T<:Real} <: DiscreteUnivariateDistribution
    n::Int
    p::T

    Binomial{T}(n, p) where {T <: Real} = new{T}(n, p)
end

function Binomial(n::Integer, p::Real; check_args::Bool=true)
    @check_args Binomial (n, n >= zero(n)) (p, zero(p) <= p <= one(p))
    return Binomial{typeof(p)}(n, p)
end

Binomial(n::Integer, p::Integer; check_args::Bool=true) = Binomial(n, float(p); check_args=check_args)
function Binomial(n::Integer; check_args::Bool=true)
    @check_args Binomial (n, n >= zero(n))
    Binomial{Float64}(n, 0.5)
end
Binomial() = Binomial{Float64}(1, 0.5)

@distr_support Binomial 0 d.n

#### Conversions

function convert(::Type{Binomial{T}}, n::Int, p::Real) where T<:Real
    return Binomial(n, T(p))
end
function Base.convert(::Type{Binomial{T}}, d::Binomial) where {T<:Real}
    return Binomial{T}(d.n, T(d.p))
end
Base.convert(::Type{Binomial{T}}, d::Binomial{T}) where {T<:Real} = d

#### Parameters

ntrials(d::Binomial) = d.n
succprob(d::Binomial) = d.p
failprob(d::Binomial{T}) where {T} = one(T) - d.p

params(d::Binomial) = (d.n, d.p)
@inline partype(::Binomial{T}) where {T<:Real} = T


#### Properties

mean(d::Binomial) = ntrials(d) * succprob(d)
var(d::Binomial) = ntrials(d) * succprob(d) * failprob(d)
function mode(d::Binomial{T}) where T<:Real
    (n, p) = params(d)
    n > 0 ? floor(Int, (n + 1) * d.p) : zero(T)
end
modes(d::Binomial) = Int[mode(d)]

median(d::Binomial) = round(Int,mean(d))

function skewness(d::Binomial)
    n, p1 = params(d)
    p0 = 1 - p1
    (p0 - p1) / sqrt(n * p0 * p1)
end

function kurtosis(d::Binomial)
    n, p = params(d)
    u = p * (1 - p)
    (1 - 6u) / (n * u)
end

function entropy(d::Binomial; approx::Bool=false)
    n, p1 = params(d)
    (p1 == 0 || p1 == 1 || n == 0) && return zero(p1)
    p0 = 1 - p1
    if approx
        return (log(twoπ * n * p0 * p1) + 1) / 2
    else
        lg = log(p1 / p0)
        lp = n * log(p0)
        s = exp(lp) * lp
        for k = 1:n
           lp += log((n - k + 1) / k) + lg
           s += exp(lp) * lp
        end
        return -s
    end
end

function kldivergence(p::Binomial, q::Binomial; kwargs...)
    np = ntrials(p)
    nq = ntrials(q)
    succp = succprob(p)
    succq = succprob(q)
    res = np * kldivergence(Bernoulli{typeof(succp)}(succp), Bernoulli{typeof(succq)}(succq))
    if np == nq
        iszero(np) && return zero(res)
        return res
    elseif np > nq
        return oftype(res, Inf)
    else
        # pull some terms out of the expectation to make this more efficient:
        res += logfactorial(np) - logfactorial(nq) - (nq - np) * log1p(-succq)
        res += expectation(k -> logfactorial(nq - k) - logfactorial(np - k), p)
        return res
    end
end

#### Evaluation & Sampling

@_delegate_statsfuns Binomial binom n p

function rand(rng::AbstractRNG, d::Binomial)
    p, n = d.p, d.n
    if p <= 0.5
        r = p
    else
        r = 1.0-p
    end
    if r*n <= 10.0
        y = rand(rng, BinomialGeomSampler(n,r))
    else
        y = rand(rng, BinomialTPESampler(n,r))
    end
    p <= 0.5 ? y : n-y
end

function mgf(d::Binomial, t::Real)
    n, p = params(d)
    (one(p) - p + p * exp(t)) ^ n
end
function cgf(d::Binomial, t)
    n, p = params(d)
    n * cgf(Bernoulli{typeof(p)}(p), t)
end

function cf(d::Binomial, t::Real)
    n, p = params(d)
    (one(p) - p + p * cis(t)) ^ n
end


#### Fit model

struct BinomialStats{N<:Real} <: SufficientStats
    ns::N         # the total number of successes
    ne::N         # the number of experiments
    n::Int        # the number of trials in each experiment
end

BinomialStats(ns::Real, ne::Real, n::Integer) = BinomialStats(promote(ns, ne)..., Int(n))

function suffstats(::Type{<:Binomial}, n::Integer, x::AbstractArray{<:Integer})
    z = zero(eltype(x))
    ns = z + z # possibly widened and different from `z`, e.g., if `z = true`
    for xi in x
        0 <= xi <= n || throw(DomainError(xi, "samples must be between 0 and $n"))
        ns += xi
    end
    BinomialStats(ns, length(x), n)
end

function suffstats(::Type{<:Binomial}, n::Integer, x::AbstractArray{<:Integer}, w::AbstractArray{<:Real})
    z = zero(eltype(x)) * zero(eltype(w))
    ns = ne = z + z # possibly widened and different from `z`, e.g., if `z = true`
    for (xi, wi) in zip(x, w)
        0 <= xi <= n || throw(DomainError(xi, "samples must be between 0 and $n"))
        ns += xi * wi
        ne += wi
    end
    BinomialStats(ns, ne, n)
end

const BinomData = Tuple{Int, AbstractArray}

suffstats(::Type{T}, data::BinomData) where {T<:Binomial} = suffstats(T, data...)
suffstats(::Type{T}, data::BinomData, w::AbstractArray{<:Real}) where {T<:Binomial} = suffstats(T, data..., w)

fit_mle(::Type{T}, ss::BinomialStats) where {T<:Binomial} = T(ss.n, ss.ns / (ss.ne * ss.n))

fit_mle(::Type{T}, n::Integer, x::AbstractArray{<:Integer}) where {T<:Binomial}= fit_mle(T, suffstats(T, n, x))
fit_mle(::Type{T}, n::Integer, x::AbstractArray{<:Integer}, w::AbstractArray{<:Real}) where {T<:Binomial} = fit_mle(T, suffstats(T, n, x, w))
fit_mle(::Type{T}, data::BinomData) where {T<:Binomial} = fit_mle(T, suffstats(T, data))
fit_mle(::Type{T}, data::BinomData, w::AbstractArray{<:Real}) where {T<:Binomial} = fit_mle(T, suffstats(T, data, w))

fit(::Type{T}, data::BinomData) where {T<:Binomial} = fit_mle(T, data)
fit(::Type{T}, data::BinomData, w::AbstractArray{<:Real}) where {T<:Binomial} = fit_mle(T, data, w)

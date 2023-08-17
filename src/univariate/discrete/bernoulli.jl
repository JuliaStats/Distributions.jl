"""
    Bernoulli(p)

A *Bernoulli distribution* is parameterized by a success rate `p`, which takes value 1
with probability `p` and 0 with probability `1-p`.

```math
P(X = k) = \\begin{cases}
1 - p & \\quad \\text{for } k = 0, \\\\
p & \\quad \\text{for } k = 1.
\\end{cases}
```

```julia
Bernoulli()    # Bernoulli distribution with p = 0.5
Bernoulli(p)   # Bernoulli distribution with success rate p

params(d)      # Get the parameters, i.e. (p,)
succprob(d)    # Get the success rate, i.e. p
failprob(d)    # Get the failure rate, i.e. 1 - p
```

External links:

* [Bernoulli distribution on Wikipedia](http://en.wikipedia.org/wiki/Bernoulli_distribution)
"""
struct Bernoulli{T<:Real} <: DiscreteUnivariateDistribution
    p::T

    Bernoulli{T}(p::Real) where {T <: Real} = new{T}(p)
end

function Bernoulli(p::Real; check_args::Bool=true)
    @check_args Bernoulli (p, zero(p) <= p <= one(p))
    return Bernoulli{typeof(p)}(p)
end

Bernoulli(p::Integer; check_args::Bool=true) = Bernoulli(float(p); check_args=check_args)
Bernoulli() = Bernoulli{Float64}(0.5)

@distr_support Bernoulli false true

Base.eltype(::Type{<:Bernoulli}) = Bool

#### Conversions
convert(::Type{Bernoulli{T}}, p::Real) where {T<:Real} = Bernoulli(T(p))
Base.convert(::Type{Bernoulli{T}}, d::Bernoulli) where {T<:Real} = Bernoulli{T}(T(d.p))
Base.convert(::Type{Bernoulli{T}}, d::Bernoulli{T}) where {T<:Real} = d

#### Parameters

succprob(d::Bernoulli) = d.p
failprob(d::Bernoulli) = 1 - d.p

params(d::Bernoulli) = (d.p,)
partype(::Bernoulli{T}) where {T} = T


#### Properties

mean(d::Bernoulli) = succprob(d)
var(d::Bernoulli) =  succprob(d) * failprob(d)
skewness(d::Bernoulli) = (p0 = failprob(d); p1 = succprob(d); (p0 - p1) / sqrt(p0 * p1))
kurtosis(d::Bernoulli) = 1 / var(d) - 6


mode(d::Bernoulli) = ifelse(succprob(d) > 1/2, 1, 0)

function modes(d::Bernoulli)
    p = succprob(d)
    p < 1/2 ? [0] :
    p > 1/2 ? [1] : [0, 1]
end

median(d::Bernoulli) = ifelse(succprob(d) <= 1/2, 0, 1)

function entropy(d::Bernoulli)
    p0 = failprob(d)
    p1 = succprob(d)
    (p0 == 0 || p0 == 1) ? zero(d.p) : -(p0 * log(p0) + p1 * log(p1))
end

#### Evaluation

pdf(d::Bernoulli, x::Bool) = x ? succprob(d) : failprob(d)
pdf(d::Bernoulli, x::Real) = x == 0 ? failprob(d) :
                             x == 1 ? succprob(d) : zero(d.p)

logpdf(d::Bernoulli, x::Real) = log(pdf(d, x))

cdf(d::Bernoulli, x::Bool) = x ? one(d.p) : failprob(d)
cdf(d::Bernoulli, x::Int) = x < 0 ? zero(d.p) :
                            x < 1 ? failprob(d) : one(d.p)

ccdf(d::Bernoulli, x::Bool) = x ? zero(d.p) : succprob(d)
ccdf(d::Bernoulli, x::Int) = x < 0 ? one(d.p) :
                             x < 1 ? succprob(d) : zero(d.p)

function quantile(d::Bernoulli{T}, p::Real) where T<:Real
    0 <= p <= 1 ? (p <= failprob(d) ? zero(T) : one(T)) : T(NaN)
end
function cquantile(d::Bernoulli{T}, p::Real) where T<:Real
    0 <= p <= 1 ? (p >= succprob(d) ? zero(T) : one(T)) : T(NaN)
end

mgf(d::Bernoulli, t::Real) = failprob(d) + succprob(d) * exp(t)
function cgf(d::Bernoulli, t)
    p, = params(d)
    # log(1-p+p*exp(t))
    logaddexp(log1p(-p), t+log(p))
end
cf(d::Bernoulli, t::Real) = failprob(d) + succprob(d) * cis(t)


#### Sampling

rand(rng::AbstractRNG, d::Bernoulli) = rand(rng) <= succprob(d)

#### MLE fitting

struct BernoulliStats{C<:Real} <: SufficientStats
    cnt0::C
    cnt1::C
end

BernoulliStats(c0::Real, c1::Real) = BernoulliStats(promote(c0, c1)...)

fit_mle(::Type{T}, ss::BernoulliStats) where {T<:Bernoulli} = T(ss.cnt1 / (ss.cnt0 + ss.cnt1))

function suffstats(::Type{<:Bernoulli}, x::AbstractArray{<:Integer})
    c0 = c1 = 0
    for xi in x
        if xi == 0
            c0 += 1
        elseif xi == 1
            c1 += 1
        else
            throw(DomainError(xi, "samples must be 0 or 1"))
        end
    end
    BernoulliStats(c0, c1)
end

function suffstats(::Type{<:Bernoulli}, x::AbstractArray{<:Integer}, w::AbstractArray{<:Real})
    length(x) == length(w) || throw(DimensionMismatch("inconsistent argument dimensions"))
    z = zero(eltype(w))
    c0 = c1 = z + z # possibly widened and different from `z`, e.g., if `z = true`
    for (xi, wi) in zip(x, w)
        if xi == 0
            c0 += wi
        elseif xi == 1
            c1 += wi
        else
            throw(DomainError(xi, "samples must be 0 or 1"))
        end
    end
    BernoulliStats(c0, c1)
end

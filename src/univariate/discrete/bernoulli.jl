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

    Bernoulli{T}(p::T) where {T <: Real} = new{T}(p)
end

function Bernoulli(p::T; check_args=true) where {T <: Real}
    check_args && @check_args(Bernoulli, zero(p) <= p <= one(p))
    return Bernoulli{T}(p)
end

Bernoulli(p::Integer) = Bernoulli(float(p))
Bernoulli() = Bernoulli(0.5, check_args=false)

@distr_support Bernoulli false true

Base.eltype(::Type{<:Bernoulli}) = Bool

#### Conversions
convert(::Type{Bernoulli{T}}, p::Real) where {T<:Real} = Bernoulli(T(p))
convert(::Type{Bernoulli{T}}, d::Bernoulli{S}) where {T <: Real, S <: Real} = Bernoulli(T(d.p), check_args=false)

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
pdf(d::Bernoulli, x::Int) = x == 0 ? failprob(d) :
                            x == 1 ? succprob(d) : zero(d.p)

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
cf(d::Bernoulli, t::Real) = failprob(d) + succprob(d) * cis(t)


#### Sampling

rand(rng::AbstractRNG, d::Bernoulli) = rand(rng) <= succprob(d)

#### MLE fitting

struct BernoulliStats <: SufficientStats
    cnt0::Float64
    cnt1::Float64

    BernoulliStats(c0::Real, c1::Real) = new(Float64(c0), Float64(c1))
end

fit_mle(::Type{<:Bernoulli}, ss::BernoulliStats) = Bernoulli(ss.cnt1 / (ss.cnt0 + ss.cnt1))

function suffstats(::Type{<:Bernoulli}, x::AbstractArray{T}) where T<:Integer
    n = length(x)
    c0 = c1 = 0
    for i = 1:n
        @inbounds xi = x[i]
        if xi == 0
            c0 += 1
        elseif xi == 1
            c1 += 1
        else
            throw(DomainError())
        end
    end
    BernoulliStats(c0, c1)
end

function suffstats(::Type{<:Bernoulli}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    n = length(x)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions."))
    c0 = c1 = 0
    for i = 1:n
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        if xi == 0
            c0 += wi
        elseif xi == 1
            c1 += wi
        else
            throw(DomainError())
        end
    end
    BernoulliStats(c0, c1)
end

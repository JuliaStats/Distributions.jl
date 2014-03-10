immutable Binomial <: DiscreteUnivariateDistribution
    size::Int
    prob::Float64
    function Binomial(n::Real, p::Real)
        n >= zero(n) || error("size must be positive")
        zero(p) <= p <= one(p) || error("prob must be in [0, 1]")
        new(int(n), float64(p))
    end
end

Binomial(size::Integer) = Binomial(size, 0.5)
Binomial() = Binomial(1, 0.5)

minimum(d::Binomial) = 0
maximum(d::Binomial) = d.size

@_jl_dist_2p Binomial binom

function entropy(d::Binomial; approx::Bool=false)
    n = d.size
    p1 = d.prob

    (p1 == 0.0 || p1 == 1.0 || n == 0) && return 0.0
    p0 = 1.0 - p1

    if approx 
        return 0.5 * (log(2.0pi * n * p0 * p1) + 1.0) 
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

kurtosis(d::Binomial) = (1.0 - 6.0 * d.prob * (1.0 - d.prob)) / var(d)

mean(d::Binomial) = d.size * d.prob

var(d::Binomial) = d.size * d.prob * (1.0 - d.prob)

skewness(d::Binomial) = (1.0 - 2.0 * d.prob) / std(d)

median(d::Binomial) = iround(d.size * d.prob)

# TODO: May need to subtract 1 sometimes
mode(d::Binomial) = d.size > 0 ? iround((d.size + 1.0) * d.prob) : 0
modes(d::Binomial) = [mode(d)]

function mgf(d::Binomial, t::Real)
    p = d.prob
    (1.0 - p + p * exp(t))^d.size
end

function cf(d::Binomial, t::Real)
    p = d.prob
    (1.0 - p + p * exp(im * t))^d.size
end

modes(d::Binomial) = iround([d.size * d.prob])

### handling support

isupperbounded(d::Union(Binomial, Type{Binomial})) = true
islowerbounded(d::Union(Binomial, Type{Binomial})) = true
isbounded(d::Union(Binomial, Type{Binomial})) = true

minimum(d::Union(Binomial, Type{Binomial})) = 0
maximum(d::Binomial) = d.size
support(d::Binomial) = 0:d.size

insupport(d::Binomial, x::Real) = isinteger(x) && 0 <= x <= d.size

## Fit model

immutable BinomialStats <: SufficientStats
    ns::Float64   # the total number of successes
    ne::Float64   # the number of experiments
    n::Int        # the number of trials in each experiment

    function BinomialStats(ns::Real, ne::Real, n::Integer)
        new(float64(ns), float64(ne), int(n))
    end
end

function suffstats{T<:Integer}(::Type{Binomial}, n::Integer, x::Array{T})
    ns = zero(T)
    for xi in x
        0 <= xi <= n || throw(DomainError())
        ns += xi
    end
    BinomialStats(ns, length(x), n)    
end

function suffstats{T<:Integer}(::Type{Binomial}, n::Integer, x::Array{T}, w::Array{Float64})
    ns = 0.
    ne = 0.
    for i = 1:length(x)
        @inbounds xi = x[i]
        0 <= xi <= n || throw(DomainError())
        @inbounds wi = w[i]
        ns += xi * wi
        ne += wi
    end
    BinomialStats(ns, ne, n)   
end

suffstats{T<:Integer}(::Type{Binomial}, data::(Int, Array{T})) = suffstats(Binomial, data...)
suffstats{T<:Integer}(::Type{Binomial}, data::(Int, Array{T}), w::Array{Float64}) = suffstats(Binomial, data..., w)

fit_mle(::Type{Binomial}, ss::BinomialStats) = Binomial(ss.n, ss.ns / (ss.ne * ss.n))

fit_mle{T<:Integer}(::Type{Binomial}, n::Integer, x::Array{T}) = fit_mle(Binomial, suffstats(Binomial, n, x))
fit_mle{T<:Integer}(::Type{Binomial}, n::Integer, x::Array{T}, w::Array{Float64}) = fit_mle(Binomial, suffstats(Binomial, n, x, w))
fit_mle{T<:Integer}(::Type{Binomial}, data::(Int, Array{T})) = fit_mle(Binomial, suffstats(Binomial, data))
fit_mle{T<:Integer}(::Type{Binomial}, data::(Int, Array{T}), w::Array{Float64}) = fit_mle(Binomial, suffstats(Binomial, data, w))


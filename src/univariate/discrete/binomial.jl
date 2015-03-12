
immutable Binomial <: DiscreteUnivariateDistribution
    n::Int
    p::Float64

    function Binomial(n::Int, p::Float64)
        n >= 0 || error("n must be non-negative.")
        0.0 <= p <= 1.0 || error("p must be in [0, 1]")
        new(n, p)
    end

    @compat Binomial(n::Integer, p::Real) = Binomial(round(Int, n), Float64(p))
    @compat Binomial(n::Integer) = Binomial(round(Int, n), 0.5)
    Binomial() = new(1, 0.5)
end

@distr_support Binomial 0 d.n

@_jl_dist_2p Binomial binom


#### Parameters

ntrials(d::Binomial) = d.n
succprob(d::Binomial) = d.p
failprob(d::Binomial) = 1.0 - d.p

params(d::Binomial) = (d.n, d.p)


#### Properties

mean(d::Binomial) = ntrials(d) * succprob(d)
var(d::Binomial) = ntrials(d) * succprob(d) * failprob(d)
mode(d::Binomial) = ((n, p) = params(d); n > 0 ? round(Int,(n + 1) * d.prob) : 0)
modes(d::Binomial) = Int[mode(d)]

median(d::Binomial) = round(Int,mean(d))

function skewness(d::Binomial) 
    n, p1 = params(d)
    p0 = 1.0 - p1
    (p0 - p1) / sqrt(n * p0 * p1)
end

function kurtosis(d::Binomial)
    n, p = params(d)
    u = p * (1.0 - p)
    (1.0 - 6.0 * u) / (n * u) 
end

function entropy(d::Binomial; approx::Bool=false)
    n, p1 = params(d)
    (p1 == 0.0 || p1 == 1.0 || n == 0) && return 0.0
    p0 = 1.0 - p1
    if approx 
        return 0.5 * (log(twoπ * n * p0 * p1) + 1.0) 
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


#### Evaluation

immutable RecursiveBinomProbEvaluator <: RecursiveProbabilityEvaluator
    n::Int
    coef::Float64   # p / (1 - p)
end

RecursiveBinomProbEvaluator(d::Binomial) = RecursiveBinomProbEvaluator(d.n, d.p / (1.0 - d.p))
nextpdf(s::RecursiveBinomProbEvaluator, p::Float64, x::Integer) = ((s.n - x + 1) / x) * s.coef * p
_pdf!(r::AbstractArray, d::Binomial, rgn::UnitRange) = _pdf!(r, d, rgn, RecursiveBinomProbEvaluator(d))


function mgf(d::Binomial, t::Real)
    n, p = params(d)
    (1.0 - p + p * exp(t)) ^ n
end

function cf(d::Binomial, t::Real)
    n, p = params(d)
    (1.0 - p + p * cis(t)) ^ n
end


#### Fit model

immutable BinomialStats <: SufficientStats
    ns::Float64   # the total number of successes
    ne::Float64   # the number of experiments
    n::Int        # the number of trials in each experiment

    function BinomialStats(ns::Real, ne::Real, n::Integer)
        @compat new(Float64(ns), Float64(ne), round(Int, n))
    end
end

function suffstats{T<:Integer}(::Type{Binomial}, n::Integer, x::AbstractArray{T})
    ns = zero(T)
    for i = 1:length(x)
        @inbounds xi = x[i]
        0 <= xi <= n || throw(DomainError())
        ns += xi
    end
    BinomialStats(ns, length(x), n)
end

function suffstats{T<:Integer}(::Type{Binomial}, n::Integer, x::AbstractArray{T}, w::AbstractArray{Float64})
    ns = 0.
    ne = 0.
    for i = 1:length(x)
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        0 <= xi <= n || throw(DomainError())        
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


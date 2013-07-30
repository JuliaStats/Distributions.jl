immutable Binomial <: DiscreteUnivariateDistribution
    size::Int
    prob::Float64
    function Binomial(n::Real, p::Real)
    	n > zero(n) || error("size must be positive")
	zero(p) <= p <= one(p) || error("prob must be in [0, 1]")
	new(int(n), float64(p))
    end
end

Binomial(size::Integer) = Binomial(size, 0.5)
Binomial() = Binomial(1, 0.5)

min(d::Binomial) = 0
max(d::Binomial) = d.size

@_jl_dist_2p Binomial binom

function entropy(d::Binomial; approx::Bool=false)
    n = d.size
    p1 = d.prob

    (p1 == 0.0 || p1 == 1.0) && return 0.0
    p0 = 1.0 - p1
    if approx return 0.5 * (log(2.0pi * n * p0 * p1) + 1.0) end
    lg = log(p1 / p0)
			# when k = 0
    lp = n * log(p0)
    s = exp(lp) * lp
    for k = 1:n
	lp += log((n - k) / (k + 1)) + lg
	s += exp(lp) * lp
    end
    -s
end

insupport(d::Binomial, x::Real) = isinteger(x) && 0 <= x <= d.size

kurtosis(d::Binomial) = (1.0 - 6.0 * d.prob * (1.0 - d.prob)) / var(d)

mean(d::Binomial) = d.size * d.prob

median(d::Binomial) = iround(d.size * d.prob)

# TODO: May need to subtract 1 sometimes
mode(d::Binomial) = iround((d.size + 1.0) * d.prob)
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

# TODO: rand() is totally screwed up

skewness(d::Binomial) = (1.0 - 2.0 * d.prob) / std(d)

var(d::Binomial) = d.size * d.prob * (1.0 - d.prob)

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
        xi = x[i]
        0 <= xi <= n || throw(DomainError())
        wi = w[i]
        ns += xi * wi
        ne += wi
    end
    BinomialStats(ns, ne, n)
end

fit_mle(::Type{Binomial}, ss::BinomialStats) = Binomial(ss.n, ss.ns / (ss.ne * ss.n))

function fit_mle{T<:Integer}(::Type{Binomial}, n::Integer, x::Array{T})
    fit_mle(Binomial, suffstats(Binomial, n, x))
end

function fit_mle{T<:Integer}(::Type{Binomial}, n::Integer, x::Array{T}, w::Array{Float64})
    fit_mle(Binomial, suffstats(Binomial, n, x, w))
end

fit(::Type{Binomial}, n::Integer, x::Array) = fit_mle(Binomial, n, x)



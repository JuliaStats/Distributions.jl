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

insupport(d::Binomial, x::Real) = isinteger(x) && 0 <= x <= d.size

min(d::Binomial) = 0
max(d::Binomial) = d.size

mean(d::Binomial) = d.size * d.prob

median(d::Binomial) = iround(d.size * d.prob)

# TODO: May need to subtract 1 sometimes
# possible to get two equal modes (e.g. prob=0.5, n odd)
mode(d::Binomial) = iround((d.size + 1.0) * d.prob)
modes(d::Binomial) = [mode(d)]

var(d::Binomial) = d.size * d.prob * (1.0 - d.prob)
skewness(d::Binomial) = (1.0 - 2.0 * d.prob) / std(d)
kurtosis(d::Binomial) = (1.0 - 6.0 * d.prob * (1.0 - d.prob)) / var(d)

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

@_jl_dist_2p Binomial binom

# Based on:
#   Catherine Loader (2000) "Fast and accurate computation of binomial probabilities"
#   available from:
#     http://projects.scipy.org/scipy/raw-attachment/ticket/620/loader2000Fast.pdf
# Uses slightly different form for D(x;n,p) function
function pdf(d::Binomial, x::Real)
    if !insupport(d,x)
        return 0.0
    end
    n, p = d.size, d.prob
    if x == 0
        return exp(n*log1p(-p))
    elseif x == n
        return p^n
    end
    q = 1.0-p
    y = n-x
    sqrt(n/(2.0*pi*x*y))*exp((lstirling(n) - lstirling(x) - lstirling(y))
                             + x*logmxp1(n*p/x) + y*logmxp1(n*q/y))
end

function logpdf(d::Binomial, x::Real)
    if !insupport(d,x)
        return -Inf
    end
    n, p = d.size, d.prob
    q = 1.0-p
    y = n-x
    if x == 0
        return n*log1p(-p)
    elseif y ==0
        return n*log(p)
    end
    (lstirling(n) - lstirling(x) - lstirling(y)) +
    x*logmxp1(n*p/x) + y*logmxp1(n*q/y) + 0.5*(log(n/(x*y))-log2π)
end


function quantile(d::Binomial, p::Real)
    # Edgeworth approximation
    x = round(quantile(EdgeworthSum(d,1), p))
    if cdf(d,x) >= p
        # search down
        xl = x-1.0
        while cdf(d,xl) >= p
            x = xl
            xl -= 1.0
        end
    else
        # search up
        x += 1.0
        while cdf(d,x) < p
            x += 1.0
        end
    end
    x
end


function mgf(d::Binomial, t::Real)
    p = d.prob
    (1.0 - p + p * exp(t))^d.size
end

function cf(d::Binomial, t::Real)
    p = d.prob
    (1.0 - p + p * exp(im * t))^d.size
end

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



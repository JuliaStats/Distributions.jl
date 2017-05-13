doc"""
    Geometric(p)

A *Geometric distribution* characterizes the number of failures before the first success in a sequence of independent Bernoulli trials with success rate `p`.

$P(X = k) = p (1 - p)^k, \quad \text{for } k = 0, 1, 2, \ldots.$

```julia
Geometric()    # Geometric distribution with success rate 0.5
Geometric(p)   # Geometric distribution with success rate p

params(d)      # Get the parameters, i.e. (p,)
succprob(d)    # Get the success rate, i.e. p
failprob(d)    # Get the failure rate, i.e. 1 - p
```

External links

*  [Geometric distribution on Wikipedia](http://en.wikipedia.org/wiki/Geometric_distribution)

"""

immutable Geometric{T<:Real} <: DiscreteUnivariateDistribution
    p::T

    function (::Type{Geometric{T}}){T}(p::T)
        @check_args(Geometric, zero(p) < p < one(p))
        new{T}(p)
    end

end

Geometric{T<:Real}(p::T) = Geometric{T}(p)
Geometric() = Geometric(0.5)

@distr_support Geometric 0 Inf

### Conversions
convert{T<:Real}(::Type{Geometric{T}}, p::Real) = Geometric(T(p))
convert{T <: Real, S <: Real}(::Type{Geometric{T}}, d::Geometric{S}) = Geometric(T(d.p))

### Parameters

succprob(d::Geometric) = d.p
failprob(d::Geometric) = 1 - d.p
params(d::Geometric) = (d.p,)
@inline partype{T<:Real}(d::Geometric{T}) = T


### Statistics

mean(d::Geometric) = failprob(d) / succprob(d)

median(d::Geometric) = -fld(logtwo, log1p(-d.p)) - 1

mode{T<:Real}(d::Geometric{T}) = zero(T)

var(d::Geometric) = (1 - d.p) / abs2(d.p)

skewness(d::Geometric) = (2 - d.p) / sqrt(1 - d.p)

kurtosis(d::Geometric) = 6 + abs2(d.p) / (1 - d.p)

entropy(d::Geometric) = (-xlogx(succprob(d)) - xlogx(failprob(d))) / d.p


### Evaluations

function pdf{T<:Real}(d::Geometric{T}, x::Int)
    if x >= 0
        p = d.p
        return p < one(p) / 10 ? p * exp(log1p(-p) * x) : d.p * (one(p) - p)^x
    else
        return zero(T)
    end
end

function logpdf{T<:Real}(d::Geometric{T}, x::Int)
    x >= 0 ? log(d.p) + log1p(-d.p) * x : -T(Inf)
end

immutable RecursiveGeomProbEvaluator <: RecursiveProbabilityEvaluator
    p0::Float64
end

RecursiveGeomProbEvaluator(d::Geometric) = RecursiveGeomProbEvaluator(failprob(d))
nextpdf(s::RecursiveGeomProbEvaluator, p::Real, x::Integer) = p * s.p0
_pdf!(r::AbstractArray, d::Geometric, rgn::UnitRange) = _pdf!(r, d, rgn, RecursiveGeomProbEvaluator(d))


function cdf{T<:Real}(d::Geometric{T}, x::Int)
    x < 0 && return zero(T)
    p = succprob(d)
    n = x + 1
    p < 1/2 ? -expm1(log1p(-p)*n) : 1 - (1 - p)^n
end

function ccdf{T<:Real}(d::Geometric{T}, x::Int)
    x < 0 && return one(T)
    p = succprob(d)
    n = x + 1
    p < 1/2 ? exp(log1p(-p)*n) : (1 - p)^n
end

function logcdf{T<:Real}(d::Geometric{T}, x::Int)
    x < 0 ? -T(Inf) : log1mexp(log1p(-d.p) * (x + 1))
end

logccdf(d::Geometric, x::Int) =  x < 0 ? zero(d.p) : log1p(-d.p) * (x + 1)

quantile(d::Geometric, p::Real) = invlogccdf(d, log1p(-p))

cquantile(d::Geometric, p::Real) = invlogccdf(d, log(p))

invlogcdf(d::Geometric, lp::Real) = invlogccdf(d, log1mexp(lp))

function invlogccdf{T<:Real}(d::Geometric{T}, lp::Real)
    if (lp > zero(d.p)) || isnan(lp)
        return T(NaN)
    elseif isinf(lp)
        return T(Inf)
    elseif lp == zero(d.p)
        return zero(T)
    end
    max(ceil(lp/log1p(-d.p)) - 1, zero(T))
end

function mgf(d::Geometric, t::Real)
    p = succprob(d)
    p / (expm1(-t) + p)
end

function cf(d::Geometric, t::Real)
    p = succprob(d)
    # replace with expm1 when complex version available
    p / (exp(-t*im) - 1 + p)
end


### Sampling

rand(d::Geometric) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Geometric) = floor(Int,-randexp(rng) / log1p(-d.p))


### Model Fitting

immutable GeometricStats <: SufficientStats
    sx::Float64
    tw::Float64

    GeometricStats(sx::Real, tw::Real) = new(sx, tw)
end

suffstats{T<:Integer}(::Type{Geometric}, x::AbstractArray{T}) = GeometricStats(sum(x), length(x))

function suffstats{T<:Integer}(::Type{Geometric}, x::AbstractArray{T}, w::AbstractArray{Float64})
    n = length(x)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    sx = 0.
    tw = 0.
    for i = 1:n
        wi = w[i]
        sx += wi * x[i]
        tw += wi
    end
    GeometricStats(sx, tw)
end

fit_mle(::Type{Geometric}, ss::GeometricStats) = Geometric(1 / (ss.sx / ss.tw + 1))

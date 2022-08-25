"""
    Geometric(p)

A *Geometric distribution* characterizes the number of failures before the first success in a sequence of independent Bernoulli trials with success rate `p`.

```math
P(X = k) = p (1 - p)^k, \\quad \\text{for } k = 0, 1, 2, \\ldots.
```

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
struct Geometric{T<:Real} <: DiscreteUnivariateDistribution
    p::T

    function Geometric{T}(p::T) where {T <: Real}
        new{T}(p)
    end
end

function Geometric(p::Real; check_args::Bool=true)
    @check_args Geometric (p, zero(p) < p < one(p))
    return Geometric{typeof(p)}(p)
end

Geometric() = Geometric{Float64}(0.5)

@distr_support Geometric 0 Inf

### Conversions
convert(::Type{Geometric{T}}, p::Real) where {T<:Real} = Geometric(T(p))
Base.convert(::Type{Geometric{T}}, d::Geometric) where {T<:Real} = Geometric{T}(T(d.p))
Base.convert(::Type{Geometric{T}}, d::Geometric{T}) where {T<:Real} = d

### Parameters

succprob(d::Geometric) = d.p
failprob(d::Geometric) = 1 - d.p
params(d::Geometric) = (d.p,)
partype(::Geometric{T}) where {T<:Real} = T


### Statistics

mean(d::Geometric) = failprob(d) / succprob(d)

median(d::Geometric) = -fld(logtwo, log1p(-d.p)) - 1

mode(d::Geometric{T}) where {T<:Real} = zero(T)

var(d::Geometric) = (1 - d.p) / abs2(d.p)

skewness(d::Geometric) = (2 - d.p) / sqrt(1 - d.p)

kurtosis(d::Geometric) = 6 + abs2(d.p) / (1 - d.p)

entropy(d::Geometric) = (-xlogx(succprob(d)) - xlogx(failprob(d))) / d.p

function kldivergence(p::Geometric, q::Geometric)
    x = succprob(p)
    y = succprob(q)
    if x == y
        return zero(float(x / y))
    elseif isone(x)
        return -log(y / x)
    else
        return log(x) - log(y) + (inv(x) - one(x)) * (log1p(-x) - log1p(-y))
    end
end


### Evaluations

function logpdf(d::Geometric, x::Real)
    insupport(d, x) ? log(d.p) + log1p(-d.p) * x : log(zero(d.p))
end

function cdf(d::Geometric, x::Int)
    p = succprob(d)
    n = max(x + 1, 0)
    p < 1/2 ? -expm1(log1p(-p)*n) : 1 - (1 - p)^n
end

ccdf(d::Geometric, x::Real) = ccdf_int(d, x)
function ccdf(d::Geometric, x::Int)
    p = succprob(d)
    n = max(x + 1, 0)
    p < 1/2 ? exp(log1p(-p)*n) : (1 - p)^n
end

logcdf(d::Geometric, x::Real) = logcdf_int(d, x)
logcdf(d::Geometric, x::Int) = log1mexp(log1p(-d.p) * max(x + 1, 0))

logccdf(d::Geometric, x::Real) = logccdf_int(d, x)
logccdf(d::Geometric, x::Int) =  log1p(-d.p) * max(x + 1, 0)

quantile(d::Geometric, p::Real) = invlogccdf(d, log1p(-p))

cquantile(d::Geometric, p::Real) = invlogccdf(d, log(p))

invlogcdf(d::Geometric, lp::Real) = invlogccdf(d, log1mexp(lp))

function invlogccdf(d::Geometric{T}, lp::Real) where T<:Real
    if (lp > zero(d.p)) || isnan(lp)
        return T(NaN)
    elseif isinf(lp)
        return T(Inf)
    elseif lp == zero(d.p)
        return zero(T)
    end
    max(ceil(lp/log1p(-d.p)) - 1, zero(T))
end

function laplace_transform(d::Geometric, t)
    p = succprob(d)
    p / (p - (1 - p) * expm1(-t))
end
mgf(d::Geometric, t::Real) = laplace_transform(d, -t)
function cgf(d::Geometric, t)
    p = succprob(d)
    # log(p / (1 - (1-p) * exp(t)))
    log(p) - log1mexp(t + log1p(-p))
end
cf(d::Geometric, t::Real) = laplace_transform(d, -t*im)

### Sampling

rand(rng::AbstractRNG, d::Geometric) = floor(Int,-randexp(rng) / log1p(-d.p))

### Model Fitting

struct GeometricStats <: SufficientStats
    sx::Float64
    tw::Float64

    GeometricStats(sx::Real, tw::Real) = new(sx, tw)
end

suffstats(::Type{<:Geometric}, x::AbstractArray{T}) where {T<:Integer} = GeometricStats(sum(x), length(x))

function suffstats(::Type{<:Geometric}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    n = length(x)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions."))
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

fit_mle(::Type{<:Geometric}, ss::GeometricStats) = Geometric(1 / (ss.sx / ss.tw + 1))

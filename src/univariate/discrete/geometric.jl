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

function Geometric(p::T; check_args=true) where {T <: Real}
    check_args && @check_args(Geometric, zero(p) < p < one(p))
    return Geometric{T}(p)
end

Geometric() = Geometric(0.5, check_args=false)

@distr_support Geometric 0 Inf

### Conversions
convert(::Type{Geometric{T}}, p::Real) where {T<:Real} = Geometric(T(p))
convert(::Type{Geometric{T}}, d::Geometric{S}) where {T <: Real, S <: Real} = Geometric(T(d.p), check_args=false)

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


### Evaluations

function pdf(d::Geometric{T}, x::Int) where T<:Real
    if x >= 0
        p = d.p
        return p < one(p) / 10 ? p * exp(log1p(-p) * x) : d.p * (one(p) - p)^x
    else
        return zero(T)
    end
end

function logpdf(d::Geometric{T}, x::Int) where T<:Real
    x >= 0 ? log(d.p) + log1p(-d.p) * x : -T(Inf)
end

struct RecursiveGeomProbEvaluator <: RecursiveProbabilityEvaluator
    p0::Float64
end

RecursiveGeomProbEvaluator(d::Geometric) = RecursiveGeomProbEvaluator(failprob(d))
nextpdf(s::RecursiveGeomProbEvaluator, p::Real, x::Integer) = p * s.p0

Base.broadcast!(::typeof(pdf), r::AbstractArray, d::Geometric, rgn::UnitRange) =
    _pdf!(r, d, rgn, RecursiveGeomProbEvaluator(d))
function Base.broadcast(::typeof(pdf), d::Geometric, X::UnitRange)
    r = similar(Array{promote_type(partype(d), eltype(X))}, axes(X))
    r .= pdf.(Ref(d),X)
end



function cdf(d::Geometric{T}, x::Int) where T<:Real
    x < 0 && return zero(T)
    p = succprob(d)
    n = x + 1
    p < 1/2 ? -expm1(log1p(-p)*n) : 1 - (1 - p)^n
end

function ccdf(d::Geometric{T}, x::Int) where T<:Real
    x < 0 && return one(T)
    p = succprob(d)
    n = x + 1
    p < 1/2 ? exp(log1p(-p)*n) : (1 - p)^n
end

function logcdf(d::Geometric{T}, x::Int) where T<:Real
    x < 0 ? -T(Inf) : log1mexp(log1p(-d.p) * (x + 1))
end

logccdf(d::Geometric, x::Int) =  x < 0 ? zero(d.p) : log1p(-d.p) * (x + 1)

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

"""
    truncated(d0::UnivariateDistribution; [lower::Real], [upper::Real])
    truncated(d0::UnivariateDistribution, lower::Real, upper::Real)

A _truncated distribution_ `d` of a distribution `d0` to the interval
``[l, u]=```[lower, upper]` has the probability density (mass) function:

```math
f(x; d_0, l, u) = \\frac{f_{d_0}(x)}{P_{Z \\sim d_0}(l \\le Z \\le u)}, \\quad x \\in [l, u],
```
where ``f_{d_0}(x)`` is the probability density (mass) function of ``d_0``.

The function throws an error if ``l > u``.

```julia
truncated(d0; lower=l)           # d0 left-truncated to the interval [l, Inf)
truncated(d0; upper=u)           # d0 right-truncated to the interval (-Inf, u]
truncated(d0; lower=l, upper=u)  # d0 truncated to the interval [l, u]
truncated(d0, l, u)              # d0 truncated to the interval [l, u]
```

The function falls back to constructing a [`Truncated`](@ref) wrapper.

# Implementation

To implement a specialized truncated form for distributions of type `D`, one or more of the
following methods should be implemented:
- `truncated(d0::D, l::T, u::T) where {T <: Real}`: interval-truncated
- `truncated(d0::D, ::Nothing, u::Real)`: right-truncated
- `truncated(d0::D, l::Real, u::Nothing)`: left-truncated
"""
function truncated end
function truncated(d::UnivariateDistribution, l::Real, u::Real)
    return truncated(d, promote(l, u)...)
end
function truncated(
    d::UnivariateDistribution;
    lower::Union{Real,Nothing}=nothing,
    upper::Union{Real,Nothing}=nothing,
)
    return truncated(d, lower, upper)
end
function truncated(d::UnivariateDistribution, ::Nothing, u::Real)
    # (log)ucdf = (log)tp = (log) P(X ≤ u) where X ~ d
    logucdf = logtp = logcdf(d, u)
    ucdf = tp = exp(logucdf)

    Truncated(d, promote(oftype(float(u), -Inf), u, oftype(ucdf, -Inf), zero(ucdf), ucdf, tp, logtp)...)
end
function truncated(d::UnivariateDistribution, l::Real, ::Nothing)
    # (log)lcdf = (log) P(X < l) where X ~ d
    loglcdf = if value_support(typeof(d)) === Discrete
        logsubexp(logcdf(d, l), logpdf(d, l))
    else
        logcdf(d, l)
    end
    lcdf = exp(loglcdf)

    # (log)tp = (log) P(l ≤ X) where X ∼ d
    logtp = log1mexp(loglcdf)
    tp = exp(logtp)

    Truncated(d, promote(l, oftype(float(l), Inf), loglcdf, lcdf, one(lcdf), tp, logtp)...)
end
truncated(d::UnivariateDistribution, ::Nothing, ::Nothing) = d
function truncated(d::UnivariateDistribution, l::T, u::T) where {T <: Real}
    l <= u || error("the lower bound must be less or equal than the upper bound")

    # (log)lcdf = (log) P(X < l) where X ~ d
    loglcdf = if value_support(typeof(d)) === Discrete
        logsubexp(logcdf(d, l), logpdf(d, l))
    else
        logcdf(d, l)
    end
    lcdf = exp(loglcdf)

    # (log)ucdf = (log) P(X ≤ u) where X ~ d
    logucdf = logcdf(d, u)
    ucdf = exp(logucdf)

    # (log)tp = (log) P(l ≤ X ≤ u) where X ∼ d
    logtp = logsubexp(loglcdf, logucdf)
    tp = exp(logtp)

    Truncated(d, promote(l, u, loglcdf, lcdf, ucdf, tp, logtp)...)
end

"""
    Truncated

Generic wrapper for a truncated distribution.
"""
struct Truncated{D<:UnivariateDistribution, S<:ValueSupport, T <: Real} <: UnivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::T      # lower bound
    upper::T      # upper bound
    loglcdf::T    # log-cdf of lower bound (exclusive): log P(X < lower)
    lcdf::T       # cdf of lower bound (exclusive): P(X < lower)
    ucdf::T       # cdf of upper bound (inclusive): P(X ≤ upper)

    tp::T         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::T      # log(tp), i.e. log(ucdf - lcdf)

    function Truncated(d::UnivariateDistribution, l::T, u::T, loglcdf::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real}
        new{typeof(d), value_support(typeof(d)), T}(d, l, u, loglcdf, lcdf, ucdf, tp, logtp)
    end
end

### Constructors of `Truncated` are deprecated - users should call `truncated`
@deprecate Truncated(d::UnivariateDistribution, l::Real, u::Real) truncated(d, l, u)
@deprecate Truncated(d::UnivariateDistribution, l::T, u::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real} Truncated(d, l, u, log(lcdf), lcdf, ucdf, tp, logtp)

params(d::Truncated) = tuple(params(d.untruncated)..., d.lower, d.upper)
partype(d::Truncated) = partype(d.untruncated)
Base.eltype(::Type{Truncated{D, S, T} } ) where {D, S, T} = T

### range and support

islowerbounded(d::Truncated) = islowerbounded(d.untruncated) || isfinite(d.lower)
isupperbounded(d::Truncated) = isupperbounded(d.untruncated) || isfinite(d.upper)

minimum(d::Truncated) = max(minimum(d.untruncated), d.lower)
maximum(d::Truncated) = min(maximum(d.untruncated), d.upper)

function insupport(d::Truncated{D,<:Union{Discrete,Continuous}}, x::Real) where {D<:UnivariateDistribution}
    return d.lower <= x <= d.upper && insupport(d.untruncated, x)
end

### evaluation

quantile(d::Truncated, p::Real) = quantile(d.untruncated, d.lcdf + p * d.tp)

function pdf(d::Truncated, x::Real)
    result = pdf(d.untruncated, x) / d.tp
    return d.lower <= x <= d.upper ? result : zero(result)
end

function logpdf(d::Truncated, x::Real)
    result = logpdf(d.untruncated, x) - d.logtp
    return d.lower <= x <= d.upper ? result : oftype(result, -Inf)
end

function cdf(d::Truncated, x::Real)
    result = (cdf(d.untruncated, x) - d.lcdf) / d.tp
    return if x < d.lower
        zero(result)
    elseif x >= d.upper
        one(result)
    else
        result
    end
end

function logcdf(d::Truncated, x::Real)
    result = logsubexp(logcdf(d.untruncated, x), d.loglcdf) - d.logtp
    return if x < d.lower
        oftype(result, -Inf)
    elseif x >= d.upper
        zero(result)
    else
        result
    end
end

function ccdf(d::Truncated, x::Real)
    result = (d.ucdf - cdf(d.untruncated, x)) / d.tp
    return if x <= d.lower
        one(result)
    elseif x > d.upper
        zero(result)
    else
        result
    end
end

function logccdf(d::Truncated, x::Real)
    result = logsubexp(logccdf(d.untruncated, x), log1p(-d.ucdf)) - d.logtp
    return if x <= d.lower
        zero(result)
    elseif x > d.upper
        oftype(result, -Inf)
    else
        result
    end
end

## random number generation

function rand(rng::AbstractRNG, d::Truncated)
    d0 = d.untruncated
    tp = d.tp
    if tp > 0.25
        while true
            r = rand(rng, d0)
            if d.lower <= r <= d.upper
                return r
            end
        end
    elseif tp > sqrt(eps(typeof(float(tp))))
        return quantile(d0, d.lcdf + rand(rng) * d.tp)
    else
        # computation in log-space fixes numerical issues if d.tp is small (#1548)
        return invlogcdf(d0, logaddexp(d.loglcdf, d.logtp - randexp(rng)))
    end
end

## show

function show(io::IO, d::Truncated)
    print(io, "Truncated(")
    d0 = d.untruncated
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals) :
          show_oneline(io, d0, namevals)
    if d.lower > -Inf
        if d.upper < Inf
            print(io, "; lower=$(d.lower), upper=$(d.upper))")
        else
            print(io, "; lower=$(d.lower))")
        end
    elseif d.upper < Inf
        print(io, "; upper=$(d.upper))")
    else
        print(io, ")")
    end
    uml && println(io)
end

_use_multline_show(d::Truncated) = _use_multline_show(d.untruncated)


### specialized truncated distributions

include(joinpath("truncated", "normal.jl"))
include(joinpath("truncated", "exponential.jl"))
include(joinpath("truncated", "uniform.jl"))
include(joinpath("truncated", "loguniform.jl"))
include(joinpath("truncated", "discrete_uniform.jl"))

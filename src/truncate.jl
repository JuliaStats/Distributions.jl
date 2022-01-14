"""
    truncated(d::UnivariateDistribution, l::Real, u::Real)

Truncate a univariate distribution `d` to the interval `[l, u]`.

The lower bound `l` can be finite or `-Inf` and the upper bound `u` can be finite or
`Inf`. The function throws an error if `l > u`.

The function falls back to constructing a [`Truncated`](@ref) wrapper.

# Implementation

To implement a specialized truncated form for distributions of type `D`, the method
`truncated(d::D, l::T, u::T) where {T <: Real}` should be implemented.
"""
function truncated(d::UnivariateDistribution, l::Real, u::Real)
    return truncated(d, promote(l, u)...)
end

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

    Truncated(d, promote(l, u, lcdf, ucdf, tp, logtp)...)
end

"""
    Truncated

Generic wrapper for a truncated distribution.
"""
struct Truncated{D<:UnivariateDistribution, S<:ValueSupport, T <: Real} <: UnivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::T      # lower bound
    upper::T      # upper bound
    lcdf::T       # cdf of lower bound (exclusive): P(X < lower)
    ucdf::T       # cdf of upper bound (inclusive): P(X ≤ upper)

    tp::T         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::T      # log(tp), i.e. log(ucdf - lcdf)
    function Truncated(d::UnivariateDistribution, l::T, u::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real}
        new{typeof(d), value_support(typeof(d)), T}(d, l, u, lcdf, ucdf, tp, logtp)
    end
end

### Constructors of `Truncated` are deprecated - users should call `truncated`
@deprecate Truncated(d::UnivariateDistribution, l::Real, u::Real) truncated(d, l, u)

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
    result = logsubexp(logcdf(d.untruncated, x), log(d.lcdf)) - d.logtp
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

function _rand!(rng::AbstractRNG, d::Truncated)
    d0 = d.untruncated
    if d.tp > 0.25
        while true
            r = _rand!(rng, d0)
            if d.lower <= r <= d.upper
                return r
            end
        end
    else
        return quantile(d0, d.lcdf + rand(rng) * d.tp)
    end
end


## show

function show(io::IO, d::Truncated)
    print(io, "Truncated(")
    d0 = d.untruncated
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals) :
          show_oneline(io, d0, namevals)
    print(io, ", range=($(d.lower), $(d.upper)))")
    uml && println(io)
end

_use_multline_show(d::Truncated) = _use_multline_show(d.untruncated)


### specialized truncated distributions

include(joinpath("truncated", "normal.jl"))
include(joinpath("truncated", "exponential.jl"))
include(joinpath("truncated", "uniform.jl"))
include(joinpath("truncated", "loguniform.jl"))
include(joinpath("truncated", "discrete_uniform.jl"))

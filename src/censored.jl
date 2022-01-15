"""
    censored(d::UnivariateDistribution, l::Union{Real,Missing}, u::Union{Real,Missing})

Censor a univariate distribution `d` to the interval `[l, u]`.

When sampling from a censored distribution, values sampled from the distribution exceeding an upper limit `u` or falling below a lower limit `r` are replaced by the respective limit `u` or `l`.

The density function (or mass function for discrete `d`) of the censored distribution is
```math
f(x; d, l, u) = \\begin{cases}
    P_d(x \\le l), & x = l \\\\
    f_d(x),         & l < x < u \\\\
    P_d(x \\ge u), & x = u \\\\
  \\end{cases},
```
where ``f_d(x)`` is the density (or mass) function of `d`, and ``P_d(x \\le b)`` is the
cumulative distribution function of ``d`` evaluated at ``x = b``.
If ``X`` is a random variable from ``d``, then `clamp(X, l, u)` is a random variable from
its censored version. Note that this implies that even if ``d`` is continuous, its censored
form assigns positive probability to the bounds ``l`` and `u``. Therefore a censored continuous distribution has atoms and is a mixture of a discrete and continuous components.

The lower bound `l` can be finite or `missing` and the upper bound `u` can be finite or
`missing`. The function throws an error if `l > u`.

The function falls back to constructing a [`Censored`](@ref) wrapper.

# Implementation

To implement a specialized censored form for distributions of type `D`, one or more of the
following methods should be implemented:
- `censored(d::D, l::T, u::T) where {T <: Real}`
- `censored(d::D, ::Missing, u::Real)`
- `censored(d::D, l::Real, ::Missing)`
"""
censored
function censored(d::UnivariateDistribution, l::T, u::T) where {T<:Real}
    return Censored(d, l, u)
end
function censored(d::UnivariateDistribution, ::Missing, u::Real)
    return Censored(d, missing, u)
end
function censored(d::UnivariateDistribution, l::Real, ::Missing)
    return Censored(d, l, missing)
end
censored(d::UnivariateDistribution, l::Real, u::Real) = censored(d, promote(l, u)...)
censored(d::UnivariateDistribution, ::Missing, ::Missing) = d

"""
    Censored

Generic wrapper for a [`censored`](@ref) distribution.
"""
struct Censored{
    D<:UnivariateDistribution,
    S<:ValueSupport,
    T<:Real,
    TL<:Union{T,Missing},
    TU<:Union{T,Missing},
} <: UnivariateDistribution{S}
    uncensored::D      # the original distribution (uncensored)
    lower::TL      # lower bound
    upper::TU      # upper bound
    function Censored(d::UnivariateDistribution, l::T, u::T) where {T<:Real}
        _is_non_empty_interval(l, u) ||
            error("the lower bound must be less than or equal to the upper bound")
        new{typeof(d), value_support(typeof(d)), T, T, T}(d, l, u)
    end
    function Censored(d::UnivariateDistribution, l::Missing, u::T) where {T <: Real}
        new{typeof(d), value_support(typeof(d)), T, Missing, T}(d, l, u)
    end
    function Censored(d::UnivariateDistribution, l::T, u::Missing) where {T <: Real}
        new{typeof(d), value_support(typeof(d)), T, T, Missing}(d, l, u)
    end
end

function censored(d::Censored, l::T, u::T) where {T<:Real}
    return censored(
        d.uncensored,
        d.lower === missing ? l : max(l, d.lower),
        d.upper === missing ? u : min(u, d.upper),
    )
end
function censored(d::Censored, ::Missing, u::Real)
    return censored(d.uncensored, d.lower, d.upper === missing ? u : min(u, d.upper))
end
function censored(d::Censored, l::Real, ::Missing)
    return censored(d.uncensored, d.lower === missing ? l : max(l, d.lower), d.upper)
end

function params(d::Censored)
    d0params = params(d.uncensored)
    return if d.lower === missing
        (d0params..., d.upper)
    elseif d.upper === missing
        (d0params..., d.lower)
    else
        (d0params..., d.lower, d.upper)
    end
end

partype(d::Censored) = partype(d.uncensored)

Base.eltype(::Type{<:Censored{D,S,T}}) where {D,S,T} = promote_type(T, eltype(D))

#### Range and Support

function islowerbounded(d::Censored)
    return (
        islowerbounded(d.uncensored) ||
        (d.lower !== missing && cdf(d.uncensored, d.lower) > 0)
    )
end
function isupperbounded(d::Censored)
    return (
        isupperbounded(d.uncensored) ||
        (d.upper !== missing && _ccdf_inc(d.uncensored, d.upper) > 0)
    )
end

function minimum(d::Censored)
    d0min = minimum(d.uncensored)
    return d.lower === missing ? d0min : max(d0min, d.lower)
end

function maximum(d::Censored)
    d0max = maximum(d.uncensored)
    return d.upper === missing ? d0max : min(d0max, d.upper)
end

function insupport(d::Censored{<:UnivariateDistribution}, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return (
        _in_open_interval(x, lower, upper) ||
        (_eqnotmissing(x, lower) && cdf(d0, lower) > 0) ||
        (_eqnotmissing(x, upper) && _ccdf_inc(d0, upper) > 0)
    )
end

#### Show

function show(io::IO, d::Censored)
    print(io, "Censored(")
    d0 = d.uncensored
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals; newline=false) : show_oneline(io, d0, namevals)
    print(io, ", range=(", d.lower, ", ", d.upper, "))")
end

_use_multline_show(d::Censored) = _use_multline_show(d.uncensored)


#### Statistics

quantile(d::Censored, p::Real) = _clamp(quantile(d.uncensored, p), d.lower, d.upper)

median(d::Censored) = _clamp(median(d.uncensored), d.lower, d.upper)

function mean(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    dtrunc = _to_truncated(d)
    prob_lower = lower === missing ? 0 : _cdf_noninc(d0, lower)
    prob_upper = upper === missing ? 0 : ccdf(d0, upper)
    prob_interval = 1 - (prob_lower + prob_upper)

    μ = prob_interval * mean(dtrunc)
    if !iszero(prob_lower)
        μ += prob_lower * lower
    end
    if !iszero(prob_upper)
        μ += prob_upper * upper
    end
    return μ
end

function var(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    dtrunc = _to_truncated(d)
    prob_lower = lower === missing ? 0 : _cdf_noninc(d0, lower)
    prob_upper = upper === missing ? 0 : ccdf(d0, upper)
    prob_interval = 1 - (prob_lower + prob_upper)
    μinterval = mean(dtrunc)

    μ = prob_interval * μinterval
    if !iszero(prob_lower)
        μ += prob_lower * lower
    end
    if !iszero(prob_upper)
        μ += prob_upper * upper
    end

    v = prob_interval * (var(dtrunc) + abs2(μinterval - μ))
    if !iszero(prob_lower)
        v += prob_lower * abs2(lower - μ)
    end
    if !iszero(prob_upper)
        v += prob_upper * abs2(upper - μ)
    end
    return v
end

function entropy(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    dtrunc = _to_truncated(d)
    entropy_dtrunc = entropy(dtrunc)

    llogcdf = lower === missing ? oftype(float(entropy_dtrunc), -Inf) : logcdf(d0, lower)
    prob_lower = if lower === missing
        0
    elseif value_support(typeof(d)) === Discrete
        logsubexp(llogcdf, logpdf(d0, lower))
    else
        llogcdf
    end
    prob_upper = upper === missing ? 0 : ccdf(d0, upper)
    prob_interval = 1 - (prob_lower + prob_upper)
    result = prob_interval * (entropy(dtrunc) - log(prob_interval)) -
        prob_lower * llogcdf - xlogx(prob_upper)
    return result
end


#### Evaluation

function pdf(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return if lower !== missing && x == lower
        result = cdf(d0, x)
        _eqnotmissing(x, upper) ? one(result) : result
    elseif _eqnotmissing(x, upper)
        _ccdf_inc(d0, x)
    else
        result = pdf(d0, x)
        _in_open_interval(x, lower, upper) ? result : zero(result)
    end
end

function logpdf(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return if lower !== missing && x == lower
        result = logcdf(d0, x)
        _eqnotmissing(x, upper) ? zero(result) : result
    elseif _eqnotmissing(x, upper)
        _logccdf_inc(d0, x)
    else
        result = logpdf(d0, x)
        _in_open_interval(x, lower, upper) ? result : oftype(result, -Inf)
    end

end

function loglikelihood(d::Censored, x::AbstractArray{<:Real, M}) where {M}
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    log_prob_lower = lower === missing ? 0 : logcdf(d0, lower)
    log_prob_upper = upper === missing ? 0 : _logccdf_inc(d0, upper)

    return sum(x) do xi
        _in_open_interval(xi, lower, upper) && return logpdf(d0, xi)
        _eqnotmissing(xi, lower) && return log_prob_lower
        _eqnotmissing(xi, upper) && return log_prob_upper
        T = float(Base.promote_eltype(log_prob_lower, log_prob_upper))
        return T(-Inf)
    end
end

function cdf(d::Censored, x::Real)
    result = cdf(d.uncensored, x)
    return if d.lower !== missing && x < d.lower
        zero(result)
    elseif d.upper === missing || x < d.upper
        result
    else
        one(result)
    end
end

function logcdf(d::Censored, x::Real)
    result = logcdf(d.uncensored, x)
    return if d.lower !== missing && x < d.lower
        oftype(result, -Inf)
    elseif d.upper === missing || x < d.upper
        result
    else
        zero(result)
    end
end

function ccdf(d::Censored, x::Real)
    result = ccdf(d.uncensored, x)
    return if d.lower !== missing && x < d.lower
        one(result)
    elseif d.upper === missing || x < d.upper
        result
    else
        zero(result)
    end
end

function logccdf(d::Censored, x::Real)
    result = logccdf(d.uncensored, x)
    return if d.lower !== missing && x < d.lower
        zero(result)
    elseif d.upper === missing || x < d.upper
        result
    else
        oftype(result, -Inf)
    end
end


#### Sampling

rand(rng::AbstractRNG, d::Censored) = _clamp(rand(rng, d.uncensored), d.lower, d.upper)


#### Utilities

# utilities to handle intervals represented with possibly missing bounds

@inline _is_non_empty_interval(l::Real, u::Real) = l ≤ u
@inline _is_non_empty_interval(l, u) = true

@inline _in_open_interval(x::Real, l::Real, u::Real) = l < x < u
@inline _in_open_interval(x::Real, ::Missing, u::Real) = x < u
@inline _in_open_interval(x::Real, l::Real, ::Missing) = x > l

function _to_truncated(d::Censored)
    return truncated(
        d.uncensored,
        d.lower === missing ? -Inf : d.lower,
        d.upper === missing ? Inf : d.upper,
    )
end

_clamp(x, l, u) = clamp(x, l, u)
_clamp(x, ::Missing, u) = min(x, u)
_clamp(x, l, ::Missing) = max(x, l)

@inline function _eqnotmissing(x, y)
    result = x == y
    return result === missing ? false : result
end

# utilities for non-inclusive CDF p(x < u) and inclusive CCDF (p ≥ u)
_cdf_noninc(d::UnivariateDistribution, x) = cdf(d, x)
_cdf_noninc(d::DiscreteUnivariateDistribution, x) = cdf(d, x) - pdf(d, x)

_logcdf_noninc(d::UnivariateDistribution, x) = logcdf(d, x)
_logcdf_noninc(d::DiscreteUnivariateDistribution, x) = logsubexp(logcdf(d, x), logpdf(d, x))

_ccdf_inc(d::UnivariateDistribution, x) = ccdf(d, x)
_ccdf_inc(d::DiscreteUnivariateDistribution, x) = ccdf(d, x) + pdf(d, x)

_logccdf_inc(d::UnivariateDistribution, x) = logccdf(d, x)
_logccdf_inc(d::DiscreteUnivariateDistribution, x) = logaddexp(logccdf(d, x), logpdf(d, x))

"""
    censored(d0::UnivariateDistribution, l::Union{Real,Missing}, u::Union{Real,Missing})

A _censored distribution_ `d` of a distribution `d0` to the interval `[l, u]` has the
probability density (mass) function:

```math
f(x; d_0, l, u) = \\begin{cases}
    P_{Z \\sim d_0}(Z \\le l), & x = l \\\\
    f_{d_0}(x),              & l < x < u \\\\
    P_{Z \\sim d_0}(Z \\ge u), & x = u \\\\
  \\end{cases}, \\quad x \\in [l, u]
```
where ``f_{d_0}`` is the probability density (mass) function of ``d_0``.

If ``Z`` is a variate from ``d_0``, and `X = clamp(Z, l, u)`, then ``X`` is a variate from
``d``, the censored version of ``d_0``. Note that this implies that even if ``d`` is
continuous, its censored form assigns positive probability to the bounds ``l`` and `u``.
Therefore a censored continuous distribution has atoms and is a mixture of discrete and
continuous components.

```julia
censored(d0, l, missing)   # d0 left-censored to the interval [l, Inf)
censored(d0, missing, u)   # d0 right-censored to the interval (-Inf, u]
censored(d0, l, u)         # d0 interval-censored to the interval [l, u]
```

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
    return (d0params..., d.lower, d.upper)
end

partype(d::Censored{<:UnivariateDistribution,<:ValueSupport,T}) where {T} = promote_type(partype(d.uncensored), T)

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
    return d.lower === missing ? min(d0min, d.upper) : max(d0min, d.lower)
end

function maximum(d::Censored)
    d0max = maximum(d.uncensored)
    return d.upper === missing ? max(d0max, d.lower) : min(d0max, d.upper)
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

# the expectations use the following relation:
# 𝔼_{x ~ d}[h(x)] = P_{x ~ d₀}(x < l) h(l) + P_{x ~ d₀}(x > u) h(u)
#                 + P_{x ~ d₀}(l ≤ x ≤ u) 𝔼_{x ~ τ}[h(x)],
# where d₀ is the uncensored distribution, d is d₀ censored to [l, u],
# and τ is d₀ truncated to [l, u]

function mean(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    prob_lower = lower === missing ? 0 : _cdf_noninc(d0, lower)
    prob_upper = upper === missing ? 0 : ccdf(d0, upper)
    prob_interval = 1 - (prob_lower + prob_upper)
    if iszero(prob_interval) # truncation contains no probability
        if lower === missing
            return one(prob_interval) * upper
        elseif upper === missing
            return one(prob_interval) * lower
        else
            return prob_lower * (iszero(prob_lower) ? one(lower) : lower) +
                   prob_upper * (iszero(prob_upper) ? one(upper) : upper)
        end
    end
    dtrunc = _to_truncated(d)
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
    prob_lower = lower === missing ? 0 : _cdf_noninc(d0, lower)
    prob_upper = upper === missing ? 0 : ccdf(d0, upper)
    prob_interval = 1 - (prob_lower + prob_upper)
    if iszero(prob_interval) # truncation contains no probability
        if lower === missing
            return one(prob_interval) * abs2(zero(upper))
        elseif upper === missing
            return one(prob_interval) * abs2(zero(lower))
        else
            μ = prob_lower * (iszero(prob_lower) ? oneunit(lower) : lower) +
                prob_upper * (iszero(prob_upper) ? oneunit(upper) : upper)
            v = prob_lower * abs2(iszero(prob_lower) ? oneunit(lower) : lower - μ) +
                prob_upper * abs2(iszero(prob_upper) ? oneunit(upper) : upper - μ)
            return v
        end
    end

    dtrunc = _to_truncated(d)
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

# this expectation also uses the following relation:
# 𝔼_{x ~ τ}[-log d(x)] = S[τ] - log P_{x ~ d₀}(l ≤ x ≤ u)
#   + (P_{x ~ d₀}(x = l) (log P_{x ~ d₀}(x = l) - log P_{x ~ d₀}(x ≤ l)) + 
#      P_{x ~ d₀}(x = u) (log P_{x ~ d₀}(x = u) - log P_{x ~ d₀}(x ≥ u))
#   ) / P_{x ~ d₀}(l ≤ x ≤ u),
# where S[τ] is the entropy of τ.

function entropy(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    if lower !== missing
        prob_lower_inc = cdf(d0, lower)
        pl = value_support(typeof(d0)) === Discrete ? pdf(d0, lower) : 0
        prob_lower = prob_lower_inc - pl
        entropy_lower = -xlogx(prob_lower_inc)
    else
        pl = prob_lower = entropy_lower = 0
    end
    if upper !== missing
        prob_upper = ccdf(d0, upper)
        pu = value_support(typeof(d0)) === Discrete ? pdf(d0, upper) : 0
        entropy_upper = -xlogx(prob_upper + pu)
    else
        pu = prob_upper = entropy_upper = 0
    end
    result = entropy_lower + entropy_upper
    prob_interval = 1 - (prob_lower + prob_upper)
    # truncation contains no probability
    iszero(prob_interval) && return result

    dtrunc = _to_truncated(d)
    result += prob_interval * entropy(dtrunc) - xlogx(prob_interval) + xlogx(pl) + xlogx(pu)
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

function loglikelihood(d::Censored, x::AbstractArray{<:Real})
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

@inline _eqnotmissing(x::Real, y::Real) = x == y
@inline _eqnotmissing(::Real, ::Missing) = false

# utilities for non-inclusive CDF p(x < u) and inclusive CCDF (p ≥ u)
_cdf_noninc(d::UnivariateDistribution, x) = cdf(d, x)
_cdf_noninc(d::DiscreteUnivariateDistribution, x) = cdf(d, x) - pdf(d, x)

_ccdf_inc(d::UnivariateDistribution, x) = ccdf(d, x)
_ccdf_inc(d::DiscreteUnivariateDistribution, x) = ccdf(d, x) + pdf(d, x)

_logccdf_inc(d::UnivariateDistribution, x) = logccdf(d, x)
_logccdf_inc(d::DiscreteUnivariateDistribution, x) = logaddexp(logccdf(d, x), logpdf(d, x))

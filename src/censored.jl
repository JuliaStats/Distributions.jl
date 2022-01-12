"""
    censored(d::UnivariateDistribution, l::Real, u::Real)

Censor a univariate distribution `d` to the interval `[l, u]`.

The lower bound `l` can be finite or `-Inf` and the upper bound `u` can be finite or
`Inf`. The function throws an error if `l > u`.

The function falls back to constructing a [`Censored`](@ref) wrapper.

# Implementation

To implement a specialized censored form for distributions of type `D`, the method
`censored(d::D, l::T, u::T) where {T <: Real}` should be implemented.
"""
function censored(d::UnivariateDistribution, l::Real, u::Real)
    return censored(d, promote(l, u)...)
end
censored(d::UnivariateDistribution, l::T, u::T) where {T <: Real} = Censored(d, l, u)

"""
    Censored

Generic wrapper for a censored distribution.
"""
struct Censored{D<:UnivariateDistribution, S<:ValueSupport, T <: Real} <: UnivariateDistribution{S}
    uncensored::D      # the original distribution (uncensored)
    lower::T      # lower bound
    upper::T      # upper bound
    function Censored(d::UnivariateDistribution, l::T, u::T) where {T <: Real}
        l ≤ u || error("the lower bound must be less than or equal to the upper bound")
        new{typeof(d), value_support(typeof(d)), T}(d, l, u)
    end
end
Censored{D,S,T}(l::T, u::T, params...) where {D,S,T} = Censored(D(params...), l, u)

function censored(d::Censored, l::T, u::T) where {T <: Real}
    return Censored(d.uncensored, max(l, d.lower), min(u, d.upper))
end

params(d::Censored) = (d.lower, d.upper, params(d.uncensored)...)
partype(d::Censored) = Base.promote_eltype(partype(d.uncensored), d.lower)
Base.eltype(::Type{Censored{D,S,T}}) where {D,S,T} = Base.promote_eltype(eltype(D), T)

#### Range and Support

islowerbounded(d::Censored) = islowerbounded(d.uncensored) || isfinite(d.lower)
isupperbounded(d::Censored) = isupperbounded(d.uncensored) || isfinite(d.upper)

minimum(d::Censored) = max(minimum(d.uncensored), d.lower)
maximum(d::Censored) = min(maximum(d.uncensored), d.upper)

function insupport(d::Censored{<:UnivariateDistribution}, x::Real)
    return d.lower ≤ x ≤ d.upper && insupport(d.uncensored, x)
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

quantile(d::Censored, p::Real) = clamp(quantile(d.uncensored, p), d.lower, d.upper)

median(d::Censored) = clamp(median(d.uncensored), d.lower, d.upper)

function mean(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    dtrunc = truncated(d.uncensored, lower, upper)
    prob_lower = if value_support(typeof(d)) === Discrete && isfinite(lower)
        cdf(d0, lower) - pdf(d0, lower)
    else
        cdf(d0, lower)
    end
    prob_upper = ccdf(d0, upper)
    prob_interval = 1 - (prob_lower + prob_upper)

    μ = prob_interval * mean(dtrunc)
    if isfinite(lower)
        μ += prob_lower * lower
    end
    if isfinite(upper)
        μ += prob_upper * upper
    end
    return μ
end

function var(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    dtrunc = truncated(d.uncensored, lower, upper)
    prob_lower = if value_support(typeof(d)) === Discrete && isfinite(lower)
        cdf(d0, lower) - pdf(d0, lower)
    else
        cdf(d0, lower)
    end
    prob_upper = ccdf(d0, upper)
    prob_interval = 1 - (prob_lower + prob_upper)
    μinterval = mean(dtrunc)

    μ = prob_interval * μinterval
    if isfinite(lower)
        μ += prob_lower * lower
    end
    if isfinite(upper)
        μ += prob_upper * upper
    end

    v = prob_interval * (var(dtrunc) + abs2(μinterval - μ))
    if isfinite(lower)
        v += prob_lower * abs2(lower - μ)
    end
    if isfinite(upper)
        v += prob_upper * abs2(upper - μ)
    end
    return v
end

function entropy(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    dtrunc = truncated(d.uncensored, lower, upper)
    lcdf = cdf(d0, lower)
    prob_lower = if value_support(typeof(d)) === Discrete && isfinite(lower)
        lcdf - pdf(d0, lower)
    else
        lcdf
    end
    prob_upper = ccdf(d0, upper)
    prob_interval = 1 - (prob_lower + prob_upper)
    result = prob_interval * (entropy(dtrunc) - log(prob_interval)) -
        prob_lower * log(lcdf) - xlogx(prob_upper)
    return d.lower == d.upper ? zero(result) : result
end


#### Evaluation

function pdf(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return if x == lower
        result = cdf(d0, x)
        x == upper ? one(result) : result
    elseif x == upper
        uccdf = ccdf(d0, x)
        if value_support(typeof(d)) === Discrete && isfinite(upper)
            uccdf - pdf(d0, x)
        else
            uccdf
        end
    else
        result = pdf(d0, x)
        lower < x < upper ? result : zero(result)
    end
end

function logpdf(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return if x == lower
        result = logcdf(d0, x)
        x == upper ? zero(result) : result
    elseif x == upper
        ulogccdf = logccdf(d0, x)
        if value_support(typeof(d)) === Discrete && isfinite(upper)
            logsubexp(ulogccdf, logpdf(d0, x))
        else
            ulogccdf
        end
    else
        result = logpdf(d0, x)
        lower < x < upper ? result : oftype(result, -Inf)
    end
end

function cdf(d::Censored, x::Real)
    result = cdf(d.uncensored, x)
    return if x < d.lower
        zero(result)
    elseif x < d.upper
        result
    else
        one(result)
    end
end

function logcdf(d::Censored, x::Real)
    result = logcdf(d.uncensored, x)
    return if x < d.lower
        oftype(result, -Inf)
    elseif x < d.upper
        result
    else
        zero(result)
    end
end

function ccdf(d::Censored, x::Real)
    result = ccdf(d.uncensored, x)
    return if x > d.upper
        zero(result)
    elseif x > d.lower
        result
    elseif d.lower == x == d.upper
        zero(result)
    else
        one(result)
    end
end

function logccdf(d::Censored, x::Real)
    result = logccdf(d.uncensored, x)
    return if x > d.upper
        oftype(result, -Inf)
    elseif x > d.lower
        result
    elseif d.lower == x == d.upper
        oftype(result, -Inf)
    else
        zero(result)
    end
end


#### Sampling

function rand(rng::AbstractRNG, d::Censored)
    return clamp(rand(rng, d.uncensored), d.lower, d.upper)
end

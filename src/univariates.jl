#### Domain && Support

struct RealInterval{T<:Real}
    lb::T
    ub::T
end

RealInterval(lb::Real, ub::Real) = RealInterval(promote(lb, ub)...)

Base.minimum(r::RealInterval) = r.lb
Base.maximum(r::RealInterval) = r.ub
Base.extrema(r::RealInterval) = (r.lb, r.ub)
Base.in(x::Real, r::RealInterval) = r.lb <= x <= r.ub

isbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = isupperbounded(d) && islowerbounded(d)

islowerbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = minimum(d) > -Inf
isupperbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = maximum(d) < +Inf

hasfinitesupport(d::Union{D,Type{D}}) where {D<:DiscreteUnivariateDistribution} = isbounded(d)
hasfinitesupport(d::Union{D,Type{D}}) where {D<:ContinuousUnivariateDistribution} = false

Base.:(==)(r1::RealInterval, r2::RealInterval) = r1.lb == r2.lb && r1.ub == r2.ub

"""
    params(d::UnivariateDistribution)

Return a tuple of parameters. Let `d` be a distribution of type `D`, then `D(params(d)...)`
will construct exactly the same distribution as ``d``.
"""
params(d::UnivariateDistribution)

"""
    scale(d::UnivariateDistribution)

Get the scale parameter.
"""
scale(d::UnivariateDistribution)

"""
    location(d::UnivariateDistribution)

Get the location parameter.
"""
location(d::UnivariateDistribution)

"""
    shape(d::UnivariateDistribution)

Get the shape parameter.
"""
shape(d::UnivariateDistribution)

"""
    rate(d::UnivariateDistribution)

Get the rate parameter.
"""
rate(d::UnivariateDistribution)

"""
    ncategories(d::UnivariateDistribution)

Get the number of categories.
"""
ncategories(d::UnivariateDistribution)

"""
    ntrials(d::UnivariateDistribution)

Get the number of trials.
"""
ntrials(d::UnivariateDistribution)

"""
    dof(d::UnivariateDistribution)

Get the degrees of freedom.
"""
dof(d::UnivariateDistribution)

"""
    insupport(d::UnivariateDistribution, x::Any)

When `x` is a scalar, it returns whether x is within the support of `d`
(e.g., `insupport(d, x) = minimum(d) <= x <= maximum(d)`).
When `x` is an array, it returns whether every element in x is within the support of `d`.

Generic fallback methods are provided, but it is often the case that `insupport` can be
done more efficiently, and a specialized `insupport` is thus desirable.
You should also override this function if the support is composed of multiple disjoint intervals.
"""
insupport{D<:UnivariateDistribution}(d::Union{D, Type{D}}, x::Any)

function insupport!(r::AbstractArray, d::Union{D,Type{D}}, X::AbstractArray) where D<:UnivariateDistribution
    length(r) == length(X) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for i in 1 : length(X)
        @inbounds r[i] = insupport(d, X[i])
    end
    return r
end


insupport(d::Union{D,Type{D}}, X::AbstractArray) where {D<:UnivariateDistribution} =
     insupport!(BitArray(undef, size(X)), d, X)

insupport(d::Union{D,Type{D}},x::Real) where {D<:ContinuousUnivariateDistribution} = minimum(d) <= x <= maximum(d)
insupport(d::Union{D,Type{D}},x::Real) where {D<:DiscreteUnivariateDistribution} = isinteger(x) && minimum(d) <= x <= maximum(d)

support(d::Union{D,Type{D}}) where {D<:ContinuousUnivariateDistribution} = RealInterval(minimum(d), maximum(d))
support(d::Union{D,Type{D}}) where {D<:DiscreteUnivariateDistribution} = round(Int, minimum(d)):round(Int, maximum(d))

# Type used for dispatch on finite support
# T = true or false
struct FiniteSupport{T} end

## macros to declare support

macro distr_support(D, lb, ub)
    D_has_constantbounds = (isa(ub, Number) || ub == :Inf) &&
                           (isa(lb, Number) || lb == :(-Inf))

    paramdecl = D_has_constantbounds ? :(d::Union{$D, Type{<:$D}}) : :(d::$D)

    # overall
    esc(quote
        Base.minimum($(paramdecl)) = $lb
        Base.maximum($(paramdecl)) = $ub
    end)
end


##### generic methods (fallback) #####

## sampling

# multiple univariate with pre-allocated array
# we use a function barrier since for some distributions `sampler(s)` is not type-stable:
# https://github.com/JuliaStats/Distributions.jl/pull/1281
function rand!(rng::AbstractRNG, s::Sampleable{Univariate}, A::AbstractArray{<:Real})
    return _rand!(rng, sampler(s), A)
end

function _rand!(rng::AbstractRNG, sampler::Sampleable{Univariate}, A::AbstractArray{<:Real})
    for i in eachindex(A)
        @inbounds A[i] = rand(rng, sampler)
    end
    return A
end

"""
    rand(rng::AbstractRNG, d::UnivariateDistribution)

Generate a scalar sample from `d`. The general fallback is `quantile(d, rand())`.
"""
rand(rng::AbstractRNG, d::UnivariateDistribution) = quantile(d, rand(rng))

## statistics

"""
    mean(d::UnivariateDistribution)

Compute the expectation.
"""
mean(d::UnivariateDistribution)

"""
    var(d::UnivariateDistribution)

Compute the variance. (A generic std is provided as `std(d) = sqrt(var(d))`)
"""
var(d::UnivariateDistribution)

"""
    std(d::UnivariateDistribution)

Return the standard deviation of distribution `d`, i.e. `sqrt(var(d))`.
"""
std(d::UnivariateDistribution) = sqrt(var(d))

"""
    median(d::UnivariateDistribution)

Return the median value of distribution `d`. The median is the smallest `x` in the support
of `d` for which `cdf(d, x) ≥ 1/2`.
Corresponding to this definition as 1/2-quantile, a fallback is provided calling the `quantile` function.
"""
median(d::UnivariateDistribution) = quantile(d, 1//2)

"""
    modes(d::UnivariateDistribution)

Get all modes (if this makes sense).
"""
modes(d::UnivariateDistribution) = [mode(d)]

"""
    mode(d::UnivariateDistribution)

Returns the first mode.
"""
mode(d::UnivariateDistribution)

"""
    skewness(d::UnivariateDistribution)

Compute the skewness.
"""
skewness(d::UnivariateDistribution)

"""
    entropy(d::UnivariateDistribution)

Compute the entropy value of distribution `d`.
"""
entropy(d::UnivariateDistribution)

"""
    entropy(d::UnivariateDistribution, b::Real)

Compute the entropy value of distribution `d`, w.r.t. a given base.
"""
entropy(d::UnivariateDistribution, b::Real) = entropy(d) / log(b)

"""
    isplatykurtic(d)

Return whether `d` is platykurtic (*i.e* `kurtosis(d) < 0`).
"""
isplatykurtic(d::UnivariateDistribution) = kurtosis(d) < 0.0

"""
    isleptokurtic(d)

Return whether `d` is leptokurtic (*i.e* `kurtosis(d) > 0`).
"""
isleptokurtic(d::UnivariateDistribution) = kurtosis(d) > 0.0

"""
    ismesokurtic(d)

Return whether `d` is mesokurtic (*i.e* `kurtosis(d) == 0`).
"""
ismesokurtic(d::UnivariateDistribution) = kurtosis(d) ≈ 0.0

"""
    kurtosis(d::UnivariateDistribution)

Compute the excessive kurtosis.
"""
kurtosis(d::UnivariateDistribution)

"""
    kurtosis(d::Distribution, correction::Bool)

Computes excess kurtosis by default. Proper kurtosis can be returned with correction=false
"""
function kurtosis(d::Distribution, correction::Bool)
    if correction
        return kurtosis(d)
    else
        return kurtosis(d) + 3.0
    end
end

excess(d::Distribution) = kurtosis(d)
excess_kurtosis(d::Distribution) = kurtosis(d)
proper_kurtosis(d::Distribution) = kurtosis(d, false)

"""
    mgf(d::UnivariateDistribution, t)

Evaluate the [moment-generating function](https://en.wikipedia.org/wiki/Moment-generating_function) of distribution `d` at `t`.

See also [`cgf`](@ref)
"""
mgf(d::UnivariateDistribution, t)

"""
    cgf(d::UnivariateDistribution, t)

Evaluate the [cumulant-generating function](https://en.wikipedia.org/wiki/Cumulant) of distribution `d` at `t`.

The cumulant-generating-function is the logarithm of the [moment-generating function](https://en.wikipedia.org/wiki/Moment-generating_function):
`cgf = log ∘ mgf`.
In practice, however, the right hand side may have overflow issues.

See also [`mgf`](@ref)
"""
cgf(d::UnivariateDistribution, t)

"""
    cf(d::UnivariateDistribution, t)

Evaluate the characteristic function of distribution `d`.
"""
cf(d::UnivariateDistribution, t)


#### pdf, cdf, and friends

# pdf

"""
    pdf(d::UnivariateDistribution, x::Real)

Evaluate the probability density (mass) at `x`.

See also: [`logpdf`](@ref).
"""
pdf(d::UnivariateDistribution, x::Real) = exp(logpdf(d, x))

# extract value from array of zero dimension
pdf(d::UnivariateDistribution, x::AbstractArray{<:Real,0}) = pdf(d, first(x))

"""
    logpdf(d::UnivariateDistribution, x::Real)

Evaluate the logarithm of probability density (mass) at `x`.

See also: [`pdf`](@ref).
"""
logpdf(d::UnivariateDistribution, x::Real)

# extract value from array of zero dimension
logpdf(d::UnivariateDistribution, x::AbstractArray{<:Real,0}) = logpdf(d, first(x))

# loglikelihood for `Real`
Base.@propagate_inbounds loglikelihood(d::UnivariateDistribution, x::Real) = logpdf(d, x)

"""
    cdf(d::UnivariateDistribution, x::Real)

Evaluate the cumulative probability at `x`.

See also [`ccdf`](@ref), [`logcdf`](@ref), and [`logccdf`](@ref).
"""
cdf(d::UnivariateDistribution, x::Real)

# fallback for discrete distribution:
# base computation on `cdf(d, floor(Int, x))` and handle `NaN` and `±Inf`
# this is only correct for distributions with integer-valued support but will error if
# `cdf(d, ::Int)` is not defined (so it should not return incorrect values silently)
cdf(d::DiscreteUnivariateDistribution, x::Real) = cdf_int(d, x)

"""
    ccdf(d::UnivariateDistribution, x::Real)

The complementary cumulative function evaluated at `x`, i.e. `1 - cdf(d, x)`.
"""
ccdf(d::UnivariateDistribution, x::Real) = 1 - cdf(d, x)

"""
    logcdf(d::UnivariateDistribution, x::Real)

The logarithm of the cumulative function value(s) evaluated at `x`, i.e. `log(cdf(x))`.
"""
logcdf(d::UnivariateDistribution, x::Real) = log(cdf(d, x))

"""
    logdiffcdf(d::UnivariateDistribution, x::Real, y::Real)

The natural logarithm of the difference between the cumulative density function at `x` and `y`, i.e. `log(cdf(x) - cdf(y))`.
"""
function logdiffcdf(d::UnivariateDistribution, x::Real, y::Real)
    # Promote to ensure that we don't compute logcdf in low precision and then promote
    _x, _y = promote(x, y)
    _x < _y && throw(ArgumentError("requires x >= y."))
    u = logcdf(d, _x)
    v = logcdf(d, _y)
    return u + log1mexp(v - u)
end

"""
    logccdf(d::UnivariateDistribution, x::Real)

The logarithm of the complementary cumulative function values evaluated at x, i.e. `log(ccdf(x))`.
"""
logccdf(d::UnivariateDistribution, x::Real) = log(ccdf(d, x))

"""
    quantile(d::UnivariateDistribution, q::Real)

Evaluate the (generalized) inverse cumulative distribution function at `q`.

For a given `0 ≤ q ≤ 1`, `quantile(d, q)` is the smallest value `x` in the support of `d`
for which `cdf(d, x) ≥ q`.

See also: [`cquantile`](@ref), [`invlogcdf`](@ref), and [`invlogccdf`](@ref).
"""
quantile(d::UnivariateDistribution, p::Real)

"""
    cquantile(d::UnivariateDistribution, q::Real)

The complementary quantile value, i.e. `quantile(d, 1-q)`.
"""
cquantile(d::UnivariateDistribution, p::Real) = quantile(d, 1.0 - p)

"""
    invlogcdf(d::UnivariateDistribution, lp::Real)

The (generalized) inverse function of [`logcdf`](@ref).

For a given `lp ≤ 0`, `invlogcdf(d, lp)` is the smallest value `x` in the support of `d` for
which `logcdf(d, x) ≥ lp`.
"""
invlogcdf(d::UnivariateDistribution, lp::Real) = quantile(d, exp(lp))

"""
    invlogccdf(d::UnivariateDistribution, lp::Real)

The (generalized) inverse function of [`logccdf`](@ref).

For a given `lp ≤ 0`, `invlogccdf(d, lp)` is the smallest value `x` in the support of `d`
for which `logccdf(d, x) ≤ lp`.
"""
invlogccdf(d::UnivariateDistribution, lp::Real) = quantile(d, -expm1(lp))

# gradlogpdf

gradlogpdf(d::ContinuousUnivariateDistribution, x::Real) = throw(MethodError(gradlogpdf, (d, x)))


function _pdf_fill_outside!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange)
    vl = vfirst = first(X)
    vr = vlast = last(X)
    n = vlast - vfirst + 1
    if islowerbounded(d)
        lb = minimum(d)
        if vl < lb
            vl = lb
        end
    end
    if isupperbounded(d)
        ub = maximum(d)
        if vr > ub
            vr = ub
        end
    end

    # fill left part
    if vl > vfirst
        for i = 1:(vl - vfirst)
            r[i] = 0.0
        end
    end

    # fill central part: with non-zero pdf
    fm1 = vfirst - 1
    for v = vl:vr
        r[v - fm1] = pdf(d, v)
    end

    # fill right part
    if vr < vlast
        for i = (vr-vfirst+2):n
            r[i] = 0.0
        end
    end
    return vl, vr, vfirst, vlast
end

function _pdf!(r::AbstractArray{<:Real}, d::DiscreteUnivariateDistribution, X::UnitRange)
    vl,vr, vfirst, vlast = _pdf_fill_outside!(r, d, X)

    # fill central part: with non-zero pdf
    fm1 = vfirst - 1
    for v = vl:vr
        r[v - fm1] = pdf(d, v)
    end
    return r
end


abstract type RecursiveProbabilityEvaluator end

function _pdf!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange, rpe::RecursiveProbabilityEvaluator)
    vl,vr, vfirst, vlast = _pdf_fill_outside!(r, d, X)

    # fill central part: with non-zero pdf
    if vl <= vr
        fm1 = vfirst - 1
        r[vl - fm1] = pv = pdf(d, vl)
        for v = (vl+1):vr
            r[v - fm1] = pv = nextpdf(rpe, pv, v)
        end
    end

    return r
end

### special definitions for distributions with integer-valued support

function cdf_int(d::DiscreteUnivariateDistribution, x::Real)
    # handle `NaN` and `±Inf` which can't be truncated to `Int`
    isfinite_x = isfinite(x)
    _x = isfinite_x ? x : zero(x)
    c = float(cdf(d, floor(Int, _x)))
    return if isfinite_x
        c
    elseif isnan(x)
        oftype(c, NaN)
    elseif x < 0
        zero(c)
    else
        one(c)
    end
end

function ccdf_int(d::DiscreteUnivariateDistribution, x::Real)
    # handle `NaN` and `±Inf` which can't be truncated to `Int`
    isfinite_x = isfinite(x)
    _x = isfinite_x ? x : zero(x)
    c = float(ccdf(d, floor(Int, _x)))
    return if isfinite_x
        c
    elseif isnan(x)
        oftype(c, NaN)
    elseif x < 0
        one(c)
    else
        zero(c)
    end
end

function logcdf_int(d::DiscreteUnivariateDistribution, x::Real)
    # handle `NaN` and `±Inf` which can't be truncated to `Int`
    isfinite_x = isfinite(x)
    _x = isfinite_x ? x : zero(x)
    c = float(logcdf(d, floor(Int, _x)))
    return if isfinite_x
        c
    elseif isnan(x)
        oftype(c, NaN)
    elseif x < 0
        oftype(c, -Inf)
    else
        zero(c)
    end
end

function logccdf_int(d::DiscreteUnivariateDistribution, x::Real)
    # handle `NaN` and `±Inf` which can't be truncated to `Int`
    isfinite_x = isfinite(x)
    _x = isfinite_x ? x : zero(x)
    c = float(logccdf(d, floor(Int, _x)))
    return if isfinite_x
        c
    elseif isnan(x)
        oftype(c, NaN)
    elseif x < 0
        zero(c)
    else
        oftype(c, -Inf)
    end
end

# implementation of the cdf for distributions whose support is a unitrange of integers
# note: incorrect for discrete distributions whose support includes non-integer numbers
function integerunitrange_cdf(d::DiscreteUnivariateDistribution, x::Integer)
    minimum_d, maximum_d = extrema(d)
    isfinite(minimum_d) || isfinite(maximum_d) || error("support is unbounded")

    result = if isfinite(minimum_d) && !(isfinite(maximum_d) && x >= div(minimum_d + maximum_d, 2))
        c = sum(Base.Fix1(pdf, d), minimum_d:(max(x, minimum_d)))
        x < minimum_d ? zero(c) : c
    else
        c = 1 - sum(Base.Fix1(pdf, d), (min(x + 1, maximum_d)):maximum_d)
        x >= maximum_d ? one(c) : c
    end

    return result
end

function integerunitrange_ccdf(d::DiscreteUnivariateDistribution, x::Integer)
    minimum_d, maximum_d = extrema(d)
    isfinite(minimum_d) || isfinite(maximum_d) || error("support is unbounded")

    result = if isfinite(minimum_d) && !(isfinite(maximum_d) && x >= div(minimum_d + maximum_d, 2))
        c = 1 - sum(Base.Fix1(pdf, d), minimum_d:(max(x, minimum_d)))
        x < minimum_d ? one(c) : c
    else
        c = sum(Base.Fix1(pdf, d), (min(x + 1, maximum_d)):maximum_d)
        x >= maximum_d ? zero(c) : c
    end

    return result
end

function integerunitrange_logcdf(d::DiscreteUnivariateDistribution, x::Integer)
    minimum_d, maximum_d = extrema(d)
    isfinite(minimum_d) || isfinite(maximum_d) || error("support is unbounded")

    result = if isfinite(minimum_d) && !(isfinite(maximum_d) && x >= div(minimum_d + maximum_d, 2))
        c = logsumexp(logpdf(d, y) for y in minimum_d:(max(x, minimum_d)))
        x < minimum_d ? oftype(c, -Inf) : c
    else
        c = log1mexp(logsumexp(logpdf(d, y) for y in (min(x + 1, maximum_d)):maximum_d))
        x >= maximum_d ? zero(c) : c
    end

    return result
end

function integerunitrange_logccdf(d::DiscreteUnivariateDistribution, x::Integer)
    minimum_d, maximum_d = extrema(d)
    isfinite(minimum_d) || isfinite(maximum_d) || error("support is unbounded")

    result = if isfinite(minimum_d) && !(isfinite(maximum_d) && x >= div(minimum_d + maximum_d, 2))
        c = log1mexp(logsumexp(logpdf(d, y) for y in minimum_d:(max(x, minimum_d))))
        x < minimum_d ? zero(c) : c
    else
        c = logsumexp(logpdf(d, y) for y in (min(x + 1, maximum_d)):maximum_d)
        x >= maximum_d ? oftype(c, -Inf) : c
    end

    return result
end

### macros to use StatsFuns for method implementation

macro _delegate_statsfuns(D, fpre, psyms...)
    # function names from StatsFuns
    fpdf = Symbol(fpre, "pdf")
    flogpdf = Symbol(fpre, "logpdf")
    fcdf = Symbol(fpre, "cdf")
    fccdf = Symbol(fpre, "ccdf")
    flogcdf = Symbol(fpre, "logcdf")
    flogccdf = Symbol(fpre, "logccdf")
    finvcdf = Symbol(fpre, "invcdf")
    finvccdf = Symbol(fpre, "invccdf")
    finvlogcdf = Symbol(fpre, "invlogcdf")
    finvlogccdf = Symbol(fpre, "invlogccdf")

    # parameter fields
    pargs = [Expr(:(.), :d, Expr(:quote, s)) for s in psyms]

    # output type of `quantile` etc.
    T = :($D <: DiscreteUnivariateDistribution ? Int : Real)

    return quote
        $Distributions.pdf(d::$D, x::Real) = $(fpdf)($(pargs...), x)
        $Distributions.logpdf(d::$D, x::Real) = $(flogpdf)($(pargs...), x)

        $Distributions.cdf(d::$D, x::Real) = $(fcdf)($(pargs...), x)
        $Distributions.logcdf(d::$D, x::Real) = $(flogcdf)($(pargs...), x)
        $Distributions.ccdf(d::$D, x::Real) = $(fccdf)($(pargs...), x)
        $Distributions.logccdf(d::$D, x::Real) = $(flogccdf)($(pargs...), x)

        $Distributions.quantile(d::$D, q::Real) = convert($T, $(finvcdf)($(pargs...), q))
        $Distributions.cquantile(d::$D, q::Real) = convert($T, $(finvccdf)($(pargs...), q))
        $Distributions.invlogcdf(d::$D, lq::Real) = convert($T, $(finvlogcdf)($(pargs...), lq))
        $Distributions.invlogccdf(d::$D, lq::Real) = convert($T, $(finvlogccdf)($(pargs...), lq))
    end
end

##### specific distributions #####

const discrete_distributions = [
    "bernoulli",
    "bernoullilogit",
    "betabinomial",
    "binomial",
    "dirac",
    "discreteuniform",
    "discretenonparametric",
    "categorical",
    "geometric",
    "hypergeometric",
    "negativebinomial",
    "noncentralhypergeometric",
    "poisson",
    "skellam",
    "soliton",
    "poissonbinomial"
]

const continuous_distributions = [
    "arcsine",
    "beta",
    "betaprime",
    "biweight",
    "cauchy",
    "chernoff",
    "chisq",    # Chi depends on Chisq
    "chi",
    "cosine",
    "epanechnikov",
    "exponential",
    "fdist",
    "frechet",
    "gamma", "erlang",
    "pgeneralizedgaussian", # GeneralizedGaussian depends on Gamma
    "generalizedpareto",
    "generalizedextremevalue",
    "gumbel",
    "inversegamma",
    "inversegaussian",
    "johnsonsu",
    "kolmogorov",
    "ksdist",
    "ksonesided",
    "kumaraswamy",
    "laplace",
    "levy",
    "lindley",
    "logistic",
    "noncentralbeta",
    "noncentralchisq",
    "noncentralf",
    "noncentralt",
    "normal",
    "normalcanon",
    "normalinversegaussian",
    "lognormal",    # LogNormal depends on Normal
    "logitnormal",    # LogitNormal depends on Normal
    "pareto",
    "rayleigh",
    "rician",
    "semicircle",
    "skewnormal",
    "studentizedrange",
    "symtriangular",
    "tdist",
    "triangular",
    "triweight",
    "uniform",
    "loguniform", # depends on Uniform
    "vonmises",
    "weibull",
    "skewedexponentialpower"
]

include(joinpath("univariate", "locationscale.jl"))

for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end

for dname in continuous_distributions
    include(joinpath("univariate", "continuous", "$(dname).jl"))
end

include(joinpath("univariate", "orderstatistic.jl"))

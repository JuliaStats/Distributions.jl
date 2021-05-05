#### Domain && Support

struct RealInterval
    lb::Float64
    ub::Float64

    RealInterval(lb::Real, ub::Real) = new(Float64(lb), Float64(ub))
end

minimum(r::RealInterval) = r.lb
maximum(r::RealInterval) = r.ub
extrema(r::RealInterval) = (r.lb, r.ub)
in(x::Real, r::RealInterval) = (r.lb <= Float64(x) <= r.ub)

isbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = isupperbounded(d) && islowerbounded(d)

islowerbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = minimum(d) > -Inf
isupperbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = maximum(d) < +Inf

hasfinitesupport(d::Union{D,Type{D}}) where {D<:DiscreteUnivariateDistribution} = isbounded(d)
hasfinitesupport(d::Union{D,Type{D}}) where {D<:ContinuousUnivariateDistribution} = false

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
        minimum($(paramdecl)) = $lb
        maximum($(paramdecl)) = $ub
    end)
end


##### generic methods (fallback) #####

## sampling

# multiple univariate, must allocate array
rand(rng::AbstractRNG, s::Sampleable{Univariate}, dims::Dims) =
    rand!(rng, sampler(s), Array{eltype(s)}(undef, dims))
rand(rng::AbstractRNG, s::Sampleable{Univariate,Continuous}, dims::Dims) =
    rand!(rng, sampler(s), Array{float(eltype(s))}(undef, dims))

# multiple univariate with pre-allocated array
function rand!(rng::AbstractRNG, s::Sampleable{Univariate}, A::AbstractArray)
    smp = sampler(s)
    for i in eachindex(A)
        @inbounds A[i] = rand(rng, smp)
    end
    return A
end

"""
    rand(rng::AbstractRNG, d::UnivariateDistribution)

Generate a scalar sample from `d`. The general fallback is `quantile(d, rand())`.
"""
rand(rng::AbstractRNG, d::UnivariateDistribution) = quantile(d, rand(rng))

"""
    rand!(rng::AbstractRNG, ::UnivariateDistribution, ::AbstractArray)

Sample a univariate distribution and store the results in the provided array.
"""
rand!(rng::AbstractRNG, ::UnivariateDistribution, ::AbstractArray)

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

Return the median value of distribution `d`.
"""
median(d::UnivariateDistribution) = quantile(d, 0.5)

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

Evaluate the moment generating function of distribution `d`.
"""
mgf(d::UnivariateDistribution, t)

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

"""
    logpdf(d::UnivariateDistribution, x::Real)

Evaluate the logarithm of probability density (mass) at `x`.

See also: [`pdf`](@ref).
"""
logpdf(d::UnivariateDistribution, x::Real)

"""
    cdf(d::UnivariateDistribution, x::Real)

Evaluate the cumulative probability at `x`.

See also [`ccdf`](@ref), [`logcdf`](@ref), and [`logccdf`](@ref).
"""
cdf(d::UnivariateDistribution, x::Real)
cdf(d::DiscreteUnivariateDistribution, x::Integer) = cdf(d, x, FiniteSupport{hasfinitesupport(d)})

# Discrete univariate with infinite support
function cdf(d::DiscreteUnivariateDistribution, x::Integer, ::Type{FiniteSupport{false}})
    c = 0.0
    for y = minimum(d):x
        c += pdf(d, y)
    end
    return c
end

# Discrete univariate with finite support
function cdf(d::DiscreteUnivariateDistribution, x::Integer, ::Type{FiniteSupport{true}})
    # calculate from left if x < (min + max)/2
    # (same as infinite support version)
    x <= div(minimum(d) + maximum(d),2) && return cdf(d, x, FiniteSupport{false})

    # otherwise, calculate from the right
    c = 1.0
    for y = x+1:maximum(d)
        c -= pdf(d, y)
    end
    return c
end

cdf(d::DiscreteUnivariateDistribution, x::Real) = cdf(d, floor(Int,x))
cdf(d::ContinuousUnivariateDistribution, x::Real) = throw(MethodError(cdf, (d, x)))


"""
    ccdf(d::UnivariateDistribution, x::Real)

The complementary cumulative function evaluated at `x`, i.e. `1 - cdf(d, x)`.
"""
ccdf(d::UnivariateDistribution, x::Real) = 1.0 - cdf(d, x)
ccdf(d::DiscreteUnivariateDistribution, x::Integer) = 1.0 - cdf(d, x)
ccdf(d::DiscreteUnivariateDistribution, x::Real) = ccdf(d, floor(Int,x))

"""
    logcdf(d::UnivariateDistribution, x::Real)

The logarithm of the cumulative function value(s) evaluated at `x`, i.e. `log(cdf(x))`.
"""
logcdf(d::UnivariateDistribution, x::Real) = log(cdf(d, x))
logcdf(d::DiscreteUnivariateDistribution, x::Integer) = log(cdf(d, x))
logcdf(d::DiscreteUnivariateDistribution, x::Real) = logcdf(d, floor(Int,x))

"""
    logdiffcdf(d::UnivariateDistribution, x::Real, y::Real)

The natural logarithm of the difference between the cumulative density function at `x` and `y`, i.e. `log(cdf(x) - cdf(y))`.
"""
function logdiffcdf(d::UnivariateDistribution, x::Real, y::Real)
    # Promote to ensure that we don't compute logcdf in low precision and then promote
    _x, _y = promote(x, y)
    _x <= _y && throw(ArgumentError("requires x > y."))
    u = logcdf(d, _x)
    v = logcdf(d, _y)
    return u + log1mexp(v - u)
end

"""
    logccdf(d::UnivariateDistribution, x::Real)

The logarithm of the complementary cumulative function values evaluated at x, i.e. `log(ccdf(x))`.
"""
logccdf(d::UnivariateDistribution, x::Real) = log(ccdf(d, x))
logccdf(d::DiscreteUnivariateDistribution, x::Integer) = log(ccdf(d, x))
logccdf(d::DiscreteUnivariateDistribution, x::Real) = logccdf(d, floor(Int,x))

"""
    quantile(d::UnivariateDistribution, q::Real)

Evaluate the inverse cumulative distribution function at `q`.

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

The inverse function of logcdf.
"""
invlogcdf(d::UnivariateDistribution, lp::Real) = quantile(d, exp(lp))

"""
    invlogccdf(d::UnivariateDistribution, lp::Real)

The inverse function of logccdf.
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

function _pdf!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange)
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

## loglikelihood
"""
    loglikelihood(d::UnivariateDistribution, x::Union{Real,AbstractArray})

The log-likelihood of distribution `d` with respect to all samples contained in `x`.

Here `x` can be a single scalar sample or an array of samples.
"""
loglikelihood(d::UnivariateDistribution, X::AbstractArray) = sum(x -> logpdf(d, x), X)
loglikelihood(d::UnivariateDistribution, x::Real) = logpdf(d, x)

### macros to use StatsFuns for method implementation

macro _delegate_statsfuns(D, fpre, psyms...)
    dt = eval(D)
    T = dt <: DiscreteUnivariateDistribution ? :Int : :Real

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

    esc(quote
        pdf(d::$D, x::$T) = $(fpdf)($(pargs...), x)
        logpdf(d::$D, x::$T) = $(flogpdf)($(pargs...), x)

        cdf(d::$D, x::$T) = $(fcdf)($(pargs...), x)
        ccdf(d::$D, x::$T) = $(fccdf)($(pargs...), x)
        logcdf(d::$D, x::$T) = $(flogcdf)($(pargs...), x)
        logccdf(d::$D, x::$T) = $(flogccdf)($(pargs...), x)

        quantile(d::$D, q::Real) = convert($T, $(finvcdf)($(pargs...), q))
        cquantile(d::$D, q::Real) = convert($T, $(finvccdf)($(pargs...), q))
        invlogcdf(d::$D, lq::Real) = convert($T, $(finvlogcdf)($(pargs...), lq))
        invlogccdf(d::$D, lq::Real) = convert($T, $(finvlogccdf)($(pargs...), lq))
    end)
end


##### specific distributions #####

const discrete_distributions = [
    "bernoulli",
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
    "kolmogorov",
    "ksdist",
    "ksonesided",
    "laplace",
    "levy",
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
    "semicircle",
    "skewnormal",
    "studentizedrange",
    "symtriangular",
    "tdist",
    "triangular",
    "triweight",
    "uniform",
    "vonmises",
    "weibull"
]

include(joinpath("univariate", "locationscale.jl"))

for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end

for dname in continuous_distributions
    include(joinpath("univariate", "continuous", "$(dname).jl"))
end

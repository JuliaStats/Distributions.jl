#### Domain && Support

immutable RealInterval
    lb::Float64
    ub::Float64

    @compat RealInterval(lb::Real, ub::Real) = new(Float64(lb), Float64(ub))
end

minimum(r::RealInterval) = r.lb
maximum(r::RealInterval) = r.ub
@compat in(x::Real, r::RealInterval) = (r.lb <= Float64(x) <= r.ub)

@compat isbounded{D<:UnivariateDistribution}(d::Union{D,Type{D}}) = isupperbounded(d) && islowerbounded(d)

@compat islowerbounded{D<:UnivariateDistribution}(d::Union{D,Type{D}}) = minimum(d) > -Inf
@compat isupperbounded{D<:UnivariateDistribution}(d::Union{D,Type{D}}) = maximum(d) < +Inf

@compat hasfinitesupport{D<:DiscreteUnivariateDistribution}(d::Union{D,Type{D}}) = isbounded(d)
@compat hasfinitesupport{D<:ContinuousUnivariateDistribution}(d::Union{D,Type{D}}) = false

@compat function insupport!{D<:UnivariateDistribution}(r::AbstractArray, d::Union{D,Type{D}}, X::AbstractArray)
    length(r) == length(X) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for i in 1 : length(X)
        @inbounds r[i] = insupport(d, X[i])
    end
    return r
end

@compat insupport{D<:UnivariateDistribution}(d::Union{D,Type{D}}, X::AbstractArray) =
     insupport!(BitArray(size(X)), d, X)

@compat insupport{D<:ContinuousUnivariateDistribution}(d::Union{D,Type{D}},x::Real) = minimum(d) <= x <= maximum(d)
@compat insupport{D<:DiscreteUnivariateDistribution}(d::Union{D,Type{D}},x::Real) = isinteger(x) && minimum(d) <= x <= maximum(d)

@compat support{D<:ContinuousUnivariateDistribution}(d::Union{D,Type{D}}) = RealInterval(minimum(d), maximum(d))
@compat support{D<:DiscreteUnivariateDistribution}(d::Union{D,Type{D}}) = round(Int, minimum(d)):round(Int, maximum(d))

# Type used for dispatch on finite support
# T = true or false
immutable FiniteSupport{T} end

## macros to declare support

macro distr_support(D, lb, ub)
    D_has_constantbounds = (isa(ub, Number) || ub == :Inf) &&
                           (isa(lb, Number) || lb == :(-Inf))

    @compat paramdecl = D_has_constantbounds ? :(d::Union{$D, Type{$D}}) : :(d::$D)

    # overall
    esc(quote
        minimum($(paramdecl)) = $lb
        maximum($(paramdecl)) = $ub
    end)
end


##### generic methods (fallback) #####

## sampling

rand(d::UnivariateDistribution) = quantile(d, rand())

rand!(d::UnivariateDistribution, A::AbstractArray) = _rand!(sampler(d), A)
rand(d::UnivariateDistribution, n::Int) = _rand!(sampler(d), Array(eltype(d), n))
rand(d::UnivariateDistribution, shp::Dims) = _rand!(sampler(d), Array(eltype(d), shp))

## statistics

std(d::UnivariateDistribution) = sqrt(var(d))
median(d::UnivariateDistribution) = quantile(d, 0.5)
modes(d::UnivariateDistribution) = [mode(d)]
entropy(d::UnivariateDistribution, b::Real) = entropy(d) / log(b)

isplatykurtic(d::UnivariateDistribution) = kurtosis(d) > 0.0
isleptokurtic(d::UnivariateDistribution) = kurtosis(d) < 0.0
ismesokurtic(d::UnivariateDistribution) = kurtosis(d) == 0.0

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


#### pdf, cdf, and friends

# pdf

pdf(d::DiscreteUnivariateDistribution, x::Int) = throw(MethodError(pdf, (d, x)))
pdf(d::DiscreteUnivariateDistribution, x::Integer) = pdf(d, round(Int, x))
pdf(d::DiscreteUnivariateDistribution, x::Real) = isinteger(x) ? pdf(d, round(Int, x)) : 0.0

pdf(d::ContinuousUnivariateDistribution, x::Float64) = throw(MethodError(pdf, (d, x)))
@compat pdf(d::ContinuousUnivariateDistribution, x::Real) = pdf(d, Float64(x))

# logpdf

logpdf(d::DiscreteUnivariateDistribution, x::Int) = log(pdf(d, x))
logpdf(d::DiscreteUnivariateDistribution, x::Integer) = logpdf(d, round( Int, x))
logpdf(d::DiscreteUnivariateDistribution, x::Real) = isinteger(x) ? logpdf(d, round(Int, x)) : -Inf

logpdf(d::ContinuousUnivariateDistribution, x::Float64) = log(pdf(d, x))
@compat logpdf(d::ContinuousUnivariateDistribution, x::Real) = logpdf(d, Float64(x))

# cdf
cdf(d::DiscreteUnivariateDistribution, x::Int) = cdf(d, x, FiniteSupport{hasfinitesupport(d)})

# Discrete univariate with infinite support
function cdf(d::DiscreteUnivariateDistribution, x::Int, ::Type{FiniteSupport{false}})
    c = 0.0
    for y = minimum(d):x
        c += pdf(d, y)
    end
    return c
end

# Discrete univariate with finite support
function cdf(d::DiscreteUnivariateDistribution, x::Int, ::Type{FiniteSupport{true}})
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

cdf(d::ContinuousUnivariateDistribution, x::Float64) = throw(MethodError(cdf, (d, x)))
@compat cdf(d::ContinuousUnivariateDistribution, x::Real) = cdf(d, Float64(x))

# ccdf

ccdf(d::DiscreteUnivariateDistribution, x::Int) = 1.0 - cdf(d, x)
ccdf(d::DiscreteUnivariateDistribution, x::Real) = ccdf(d, floor(Int,x))
ccdf(d::ContinuousUnivariateDistribution, x::Float64) = 1.0 - cdf(d, x)
@compat ccdf(d::ContinuousUnivariateDistribution, x::Real) = ccdf(d, Float64(x))

# logcdf

logcdf(d::DiscreteUnivariateDistribution, x::Int) = log(cdf(d, x))
logcdf(d::DiscreteUnivariateDistribution, x::Real) = logcdf(d, floor(Int,x))
logcdf(d::ContinuousUnivariateDistribution, x::Float64) = log(cdf(d, x))
@compat logcdf(d::ContinuousUnivariateDistribution, x::Real) = logcdf(d, Float64(x))

# logccdf

logccdf(d::DiscreteUnivariateDistribution, x::Int) = log(ccdf(d, x))
logccdf(d::DiscreteUnivariateDistribution, x::Real) = logccdf(d, floor(Int,x))
logccdf(d::ContinuousUnivariateDistribution, x::Float64) = log(ccdf(d, x))
@compat logccdf(d::ContinuousUnivariateDistribution, x::Real) = logccdf(d, Float64(x))

# quantile

quantile(d::UnivariateDistribution, p::Float64) = throw(MethodError(quantile, (d, p)))
@compat quantile(d::UnivariateDistribution, p::Real) = quantile(d, Float64(p))

# cquantile

cquantile(d::UnivariateDistribution, p::Float64) = quantile(d, 1.0 - p)
@compat cquantile(d::UnivariateDistribution, p::Real) = cquantile(d, Float64(p))

# invlogcdf

invlogcdf(d::UnivariateDistribution, lp::Float64) = quantile(d, exp(lp))
@compat invlogcdf(d::UnivariateDistribution, lp::Real) = invlogcdf(d, Float64(lp))

# invlogccdf

invlogccdf(d::UnivariateDistribution, lp::Float64) = quantile(d, -expm1(lp))
@compat invlogccdf(d::UnivariateDistribution, lp::Real) = invlogccdf(d, Float64(lp))

# gradlogpdf

gradlogpdf(d::ContinuousUnivariateDistribution, x::Float64) = throw(MethodError(gradlogpdf, (d, x)))
@compat gradlogpdf(d::ContinuousUnivariateDistribution, x::Real) = gradlogpdf(d, Float64(x))


# vectorized versions
for fun in [:pdf, :logpdf,
            :cdf, :logcdf,
            :ccdf, :logccdf,
            :invlogcdf, :invlogccdf,
            :quantile, :cquantile]

    _fun! = symbol(string('_', fun, '!'))
    fun! = symbol(string(fun, '!'))

    @eval begin
        function ($_fun!)(r::AbstractArray, d::UnivariateDistribution, X::AbstractArray)
            for i in 1 : length(X)
                r[i] = ($fun)(d, X[i])
            end
            return r
        end

        function ($fun!)(r::AbstractArray, d::UnivariateDistribution, X::AbstractArray)
            length(r) == length(X) ||
                throw(ArgumentError("Inconsistent array dimensions."))
            $(_fun!)(r, d, X)
        end

        ($fun)(d::UnivariateDistribution, X::AbstractArray) =
            $(_fun!)(Array(Float64, size(X)), d, X)
    end
end


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


abstract RecursiveProbabilityEvaluator

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


pdf(d::DiscreteUnivariateDistribution) = isbounded(d) ? pdf(d, minimum(d):maximum(d)) :
                                                        error("pdf(d) is not allowed when d is unbounded.")


## loglikelihood

function _loglikelihood(d::UnivariateDistribution, X::AbstractArray)
    ll = 0.0
    for i in 1:length(X)
        @inbounds ll += logpdf(d, X[i])
    end
    return ll
end

loglikelihood(d::UnivariateDistribution, X::AbstractArray) =
    _loglikelihood(d, X)


### macros to use StatsFuns for method implementation

macro _delegate_statsfuns(D, fpre, psyms...)
    dt = eval(D)
    T = dt <: DiscreteUnivariateDistribution ? :Int : :Float64

    # function names from StatsFuns
    fpdf = symbol(string(fpre, "pdf"))
    flogpdf = symbol(string(fpre, "logpdf"))
    fcdf = symbol(string(fpre, "cdf"))
    fccdf = symbol(string(fpre, "ccdf"))
    flogcdf = symbol(string(fpre, "logcdf"))
    flogccdf = symbol(string(fpre, "logccdf"))
    finvcdf = symbol(string(fpre, "invcdf"))
    finvccdf = symbol(string(fpre, "invccdf"))
    finvlogcdf = symbol(string(fpre, "invlogcdf"))
    finvlogccdf = symbol(string(fpre, "invlogccdf"))

    # parameter fields
    pargs = [Expr(:(.), :d, Expr(:quote, s)) for s in psyms]

    esc(quote
        pdf(d::$D, x::$T) = $(fpdf)($(pargs...), x)
        logpdf(d::$D, x::$T) = $(flogpdf)($(pargs...), x)

        cdf(d::$D, x::$T) = $(fcdf)($(pargs...), x)
        ccdf(d::$D, x::$T) = $(fccdf)($(pargs...), x)
        logcdf(d::$D, x::$T) = $(flogcdf)($(pargs...), x)
        logccdf(d::$D, x::$T) = $(flogccdf)($(pargs...), x)

        quantile(d::$D, q::Float64) = convert($T, $(finvcdf)($(pargs...), q))
        cquantile(d::$D, q::Float64) = convert($T, $(finvccdf)($(pargs...), q))
        invlogcdf(d::$D, lq::Float64) = convert($T, $(finvlogcdf)($(pargs...), lq))
        invlogccdf(d::$D, lq::Float64) = convert($T, $(finvlogccdf)($(pargs...), lq))
    end)
end


##### specific distributions #####

const discrete_distributions = [
    "bernoulli",
    "betabinomial",
    "binomial",
    "categorical",
    "discreteuniform",
    "geometric",
    "hypergeometric",
    "negativebinomial",
    "noncentralhypergeometric",
    "poisson",
    "skellam",
    "poissonbinomial"
]

const continuous_distributions = [
    "arcsine",
    "beta",
    "betaprime",
    "biweight",
    "cauchy",
    "chisq",    # Chi depends on Chisq
    "chi",
    "cosine",
    "epanechnikov",
    "exponential",
    "fdist",
    "frechet",
    "gamma", "erlang",
    "generalizedpareto",
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
    "pareto",
    "rayleigh",
    "symtriangular",
    "tdist",
    "triangular",
    "triweight",
    "uniform",
    "vonmises",
    "weibull"
]

for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end

for dname in continuous_distributions
    include(joinpath("univariate", "continuous", "$(dname).jl"))
end

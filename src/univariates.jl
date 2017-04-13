#### Domain && Support

immutable RealInterval
    lb::Float64
    ub::Float64

    RealInterval(lb::Real, ub::Real) = new(Float64(lb), Float64(ub))
end

minimum(r::RealInterval) = r.lb
maximum(r::RealInterval) = r.ub
in(x::Real, r::RealInterval) = (r.lb <= Float64(x) <= r.ub)

isbounded{D<:UnivariateDistribution}(d::Union{D,Type{D}}) = isupperbounded(d) && islowerbounded(d)

islowerbounded{D<:UnivariateDistribution}(d::Union{D,Type{D}}) = minimum(d) > -Inf
isupperbounded{D<:UnivariateDistribution}(d::Union{D,Type{D}}) = maximum(d) < +Inf

hasfinitesupport{D<:DiscreteUnivariateDistribution}(d::Union{D,Type{D}}) = isbounded(d)
hasfinitesupport{D<:ContinuousUnivariateDistribution}(d::Union{D,Type{D}}) = false

function insupport!{D<:UnivariateDistribution}(r::AbstractArray, d::Union{D,Type{D}}, X::AbstractArray)
    length(r) == length(X) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for i in 1 : length(X)
        @inbounds r[i] = insupport(d, X[i])
    end
    return r
end

insupport{D<:UnivariateDistribution}(d::Union{D,Type{D}}, X::AbstractArray) =
     insupport!(BitArray(size(X)), d, X)

insupport{D<:ContinuousUnivariateDistribution}(d::Union{D,Type{D}},x::Real) = minimum(d) <= x <= maximum(d)
insupport{D<:DiscreteUnivariateDistribution}(d::Union{D,Type{D}},x::Real) = isinteger(x) && minimum(d) <= x <= maximum(d)

support{D<:ContinuousUnivariateDistribution}(d::Union{D,Type{D}}) = RealInterval(minimum(d), maximum(d))
support{D<:DiscreteUnivariateDistribution}(d::Union{D,Type{D}}) = round(Int, minimum(d)):round(Int, maximum(d))

# Type used for dispatch on finite support
# T = true or false
immutable FiniteSupport{T} end

## macros to declare support

macro distr_support(D, lb, ub)
    D_has_constantbounds = (isa(ub, Number) || ub == :Inf) &&
                           (isa(lb, Number) || lb == :(-Inf))

    paramdecl = D_has_constantbounds ? :(d::Union{$D, Type{$D}}) : :(d::$D)

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
rand(d::UnivariateDistribution, n::Int) = _rand!(sampler(d), Vector{eltype(d)}(n))
rand(d::UnivariateDistribution, shp::Dims) = _rand!(sampler(d), Vector{eltype(d)}(shp))

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

pdf(d::ContinuousUnivariateDistribution, x::Real) = throw(MethodError(pdf, (d, x)))

# logpdf

logpdf(d::DiscreteUnivariateDistribution, x::Int) = log(pdf(d, x))
logpdf(d::DiscreteUnivariateDistribution, x::Integer) = logpdf(d, round( Int, x))
logpdf(d::DiscreteUnivariateDistribution, x::Real) = isinteger(x) ? logpdf(d, round(Int, x)) : -Inf

logpdf(d::ContinuousUnivariateDistribution, x::Real) = log(pdf(d, x))

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

cdf(d::ContinuousUnivariateDistribution, x::Real) = throw(MethodError(cdf, (d, x)))

# ccdf

ccdf(d::DiscreteUnivariateDistribution, x::Int) = 1.0 - cdf(d, x)
ccdf(d::DiscreteUnivariateDistribution, x::Real) = ccdf(d, floor(Int,x))
ccdf(d::ContinuousUnivariateDistribution, x::Real) = 1.0 - cdf(d, x)

# logcdf

logcdf(d::DiscreteUnivariateDistribution, x::Int) = log(cdf(d, x))
logcdf(d::DiscreteUnivariateDistribution, x::Real) = logcdf(d, floor(Int,x))
logcdf(d::ContinuousUnivariateDistribution, x::Real) = log(cdf(d, x))

# logccdf

logccdf(d::DiscreteUnivariateDistribution, x::Int) = log(ccdf(d, x))
logccdf(d::DiscreteUnivariateDistribution, x::Real) = logccdf(d, floor(Int,x))

logccdf(d::ContinuousUnivariateDistribution, x::Real) = log(ccdf(d, x))

# quantile

quantile(d::UnivariateDistribution, p::Real) = throw(MethodError(quantile, (d, p)))

# cquantile

cquantile(d::UnivariateDistribution, p::Real) = quantile(d, 1.0 - p)

# invlogcdf

invlogcdf(d::UnivariateDistribution, lp::Real) = quantile(d, exp(lp))

# invlogccdf

invlogccdf(d::UnivariateDistribution, lp::Real) = quantile(d, -expm1(lp))

# gradlogpdf

gradlogpdf(d::ContinuousUnivariateDistribution, x::Real) = throw(MethodError(gradlogpdf, (d, x)))

# vectorized versions
for fun in [:pdf, :logpdf,
            :cdf, :logcdf,
            :ccdf, :logccdf,
            :invlogcdf, :invlogccdf,
            :quantile, :cquantile]

    _fun! = Symbol('_', fun, '!')
    fun! = Symbol(fun, '!')

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
            $(_fun!)(Array{promote_type(partype(d), eltype(X))}(size(X)), d, X)
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


@compat abstract type RecursiveProbabilityEvaluator end

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

loglikelihood(d::UnivariateDistribution, X::AbstractArray) = sum(x -> logpdf(d, x), X)

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

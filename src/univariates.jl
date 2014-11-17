#### Domain && Support

immutable RealInterval
    lb::Float64
    ub::Float64

    RealInterval(lb::Real, ub::Real) = new(float64(lb), float64(ub))
end

minimum(r::RealInterval) = r.lb
maximum(r::RealInterval) = r.ub

in(x::Real, r::RealInterval) = (r.lb <= float64(x) <= r.ub)

make_support{D<:DiscreteUnivariateDistribution}(::Type{D}, lb::Real, ub::Real) = int(lb):int(ub)
make_support{D<:ContinuousUnivariateDistribution}(::Type{D}, lb::Real, ub::Real) = RealInterval(lb, ub)
 
in_value_domain{D<:DiscreteUnivariateDistribution}(::Type{D}, x::Integer) = true
in_value_domain{D<:DiscreteUnivariateDistribution}(::Type{D}, x::Real) = isinteger(x)
in_value_domain{D<:ContinuousUnivariateDistribution}(::Type{D}, x::Real) = true

isbounded(d::UnivariateDistribution) = isupperbounded(d) && islowerbounded(d)
hasfinitesupport(d::DiscreteUnivariateDistribution) = isbounded(d)
hasfinitesupport(d::ContinuousUnivariateDistribution) = false

function insupport!{D<:UnivariateDistribution}(r::AbstractArray, d::Union(D,Type{D}), X::AbstractArray)
    length(r) == length(X) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for i in 1 : length(X)
        @inbounds r[i] = insupport(d, X[i])
    end
    return r
end

insupport{D<:UnivariateDistribution}(d::Union(D,Type{D}), X::AbstractArray) = 
     insupport!(BitArray(size(X)), d, X)

## macros to declare support

macro with_bounded_support(D, lb, ub)
    common = quote
        islowerbounded(::Union($D, Type{$D})) = true
        isupperbounded(::Union($D, Type{$D})) = true
        isbounded(::Union($D, Type{$D})) = true
        minimum(d::$D) = $(lb)
        maximum(d::$D) = $(ub)
    end

    if isa(lb, Number) && isa(ub, Number)
        esc(quote
            $(common)    
            insupport(::Union($D, Type{$D}), x::Real) = in_value_domain($D, x) && ($lb <= x <= $ub)
            support(::Union($D, Type{$D})) = make_support($D, $lb, $ub)
        end)
    else
        esc(quote
            $(common)
            insupport(d::$D, x::Real) = in_value_domain($D, x) && (($lb) <= x <= ($ub))
            support(d::$D) = make_support($D, $lb, $ub)
        end)
    end
end

macro with_lowerbounded_support(D, lb)
    # this macro assumes that the support is not upper-bounded
    common = quote
        islowerbounded(::Union($D, Type{$D})) = true
        isupperbounded(::Union($D, Type{$D})) = false
        isbounded(::Union($D, Type{$D})) = false
        minimum(d::$D) = $(lb)
        maximum(d::$D) = Inf
    end

    if isa(lb, Number)
        esc(quote
            $(common)
            insupport(::Union($D, Type{$D}), x::Real) = in_value_domain($D, x) && (x >= $lb)
        end)
    else
        esc(quote
            $(common)
            insupport(d::$D, x::Real) = in_value_domain($D, x) && (x >= ($lb))
        end)
    end
end

macro with_uppperbounded_support(D, ub)
    # this macro assumes that the support is not lower-bounded
    common = quote
        islowerbounded(::Union($D, Type{$D})) = false
        isupperbounded(::Union($D, Type{$D})) = true
        isbounded(::Union($D, Type{$D})) = false
        minimum(d::$D) = -Inf
        maximum(d::$D) = $(ub)
    end

    if isa(ub, Number)
        esc(quote
            $(common)
            insupport(::Union($D, Type{$D}), x::Real) = in_value_domain($D, x) && (x <= $ub)
        end)
    else
        esc(quote
            $(common)
            insupport(d::$D, x::Real) = in_value_domain($D, x) && (x <= ($ub))
        end)
    end
end

macro with_unbounded_support(D)
    quote
        islowerbounded(::Union($D, Type{$D})) = false
        isupperbounded(::Union($D, Type{$D})) = false
        isbounded(::Union($D, Type{$D})) = false
        minimum(d::$D) = -Inf
        maximum(d::$D) = Inf
        insupport(::Union($D, Type{$D}), x::Real) = in_value_domain($D, x)
    end
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


## pdf, cdf, and friends

logpdf(d::UnivariateDistribution, x::Number) = log(pdf(d, x))
cdf(d::DiscreteUnivariateDistribution, k::Real) = sum([pdf(d,i) for i in minimum(d):k])
ccdf(d::UnivariateDistribution, q::Real) = 1.0 - cdf(d, q)
cquantile(d::UnivariateDistribution, p::Real) = quantile(d, 1.0 - p)

logcdf(d::UnivariateDistribution, q::Real) = log(cdf(d,q))
logccdf(d::UnivariateDistribution, q::Real) = log(ccdf(d,q))
invlogccdf(d::UnivariateDistribution, lp::Real) = quantile(d, -expm1(lp))
invlogcdf(d::UnivariateDistribution, lp::Real) = quantile(d, exp(lp))

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

function _pdf!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange)
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
    return r
end


abstract RecursiveProbabilityEvaluator

function _pdf!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange, rpe::RecursiveProbabilityEvaluator)
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
    if vl <= vr
        fm1 = vfirst - 1
        r[vl - fm1] = pv = pdf(d, vl)
        for v = (vl+1):vr
            r[v - fm1] = pv = nextpdf(rpe, pv, v)
        end
    end

    # fill right part
    if vr < vlast
        for i = (vr-vfirst+2):n
            r[i] = 0.0
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

##### specific distributions #####

const discrete_distributions = [ 
    "bernoulli",
    "binomial",
    "categorical",
    "discreteuniform",
    "geometric",
    "hypergeometric",
    "negativebinomial",
    "noncentralhypergeometric",
    "poisson",
    "skellam"  
]

const continuous_distributions = [
    "arcsine",
    "beta",
    "betaprime",
    "cauchy",
    "chi",
    "chisq",
    "cosine",
    "edgeworth",    
    "exponential",
    "fdist",
    "frechet",
    "gamma", "erlang",
    "gumbel",
    "inversegamma",
    "inversegaussian",
    "kolmogorov",
    "ksdist",
    "ksonesided",
    "laplace",
    "levy",
    "logistic",
    "lognormal",
    "noncentralbeta",
    "noncentralchisq",
    "noncentralf",
    "noncentralt",
    "normal",
    "normalcanon",
    "pareto",
    "rayleigh",
    "symtriangular",
    "tdist",
    "triangular",
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


##############################################################################
#
# Fallback methods, usually overridden for specific distributions
#
##############################################################################

# support handling

isbounded(d::UnivariateDistribution) = isupperbounded(d) && islowerbounded(d)
hasfinitesupport(d::DiscreteUnivariateDistribution) = isbounded(d)
hasfinitesupport(d::ContinuousUnivariateDistribution) = false

insupport(d::Distribution, x) = false

function insupport!{D<:UnivariateDistribution}(r::AbstractArray, d::Union(D,Type{D}), X::AbstractArray)
    if length(r) != length(X)
        throw(ArgumentError("Inconsistent array dimensions."))
    end
    for i in 1 : length(X)
        r[i] = insupport(d, X[i])
    end
    r
end

insupport{D<:UnivariateDistribution}(d::Union(D,Type{D}), X::AbstractArray) = insupport!(BitArray(size(X)), d, X)

function insupport!{D<:MultivariateDistribution}(r::AbstractArray, d::Union(D,Type{D}), X::AbstractArray)
    n = div(length(X),size(X,1))
    if length(r) != n
        throw(ArgumentError("Inconsistent array dimensions."))
    end    
    for i in 1 : n
        r[i] = insupport(d, X[:, i])
    end
    r
end
insupport{D<:MultivariateDistribution}(d::Union(D,Type{D}), X::AbstractArray) = insupport!(BitArray(size(X)[2:end]), d, X)

function insupport!{D<:MatrixDistribution}(r::AbstractArray, d::Union(D,Type{D}), X::AbstractArray)
    n = div(length(X),size(X,1)*size(X,2))
    if length(r) != n
        throw(ArgumentError("Inconsistent array dimensions."))
    end    
    for i in 1 : n
        r[i] = insupport(d, X[:, :, i])
    end
    r
end
insupport{D<:MatrixDistribution}(d::Union(D,Type{D}), X::AbstractArray) = insupport!(BitArray(size(X)[3:end]), d, X)

#### Statistics ####
mean(d::Distribution) = throw(MethodError(mean,(d,)))
std(d::Distribution) = sqrt(var(d))

# What's the purpose for this function?
function var{M <: Real}(d::UnivariateDistribution, mu::AbstractArray{M})
    V = similar(mu, Float64)
    for i in 1:length(mu)
        V[i] = var(d, mu[i])
    end
    return V
end

function cor(d::MultivariateDistribution)
    R = cov(d)
    m, n = size(R)
    for j in 1:n
        for i in 1:n
            R[i, j] = d.cov[i, j] / sqrt(d.cov[i, i] * d.cov[j, j])
        end
    end
    return R
end

binaryentropy(d::Distribution) = entropy(d) / log(2)

# kurtosis returns excess kurtosis by default
# proper kurtosis can be returned with correction = false
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

isplatykurtic(d::Distribution) = kurtosis(d) > 0.0
isleptokurtic(d::Distribution) = kurtosis(d) < 0.0
ismesokurtic(d::Distribution) = kurtosis(d) == 0.0

median(d::UnivariateDistribution) = quantile(d, 0.5)

modes(d::Distribution) = [mode(d)]

#### pdf, cdf, and quantile ####

logpdf(d::UnivariateDistribution, x::Real) = log(pdf(d, x))
pdf(d::MultivariateDistribution, x::Vector) = exp(logpdf(d, x))

ccdf(d::UnivariateDistribution, q::Real) = 1.0 - cdf(d, q)
cquantile(d::UnivariateDistribution, p::Real) = quantile(d, 1.0 - p)

logcdf(d::Distribution, q::Real) = log(cdf(d,q))
logccdf(d::Distribution, q::Real) = log(ccdf(d,q))
invlogccdf(d::Distribution, lp::Real) = quantile(d, -expm1(lp))
invlogcdf(d::Distribution, lp::Real) = quantile(d, exp(lp))


#### log likelihood ####

function loglikelihood(d::UnivariateDistribution, X::Array)
    ll = 0.0
    for i in 1:length(X)
        ll += logpdf(d, X[i])
    end
    return ll
end

function loglikelihood(d::MultivariateDistribution, X::Matrix)
    ll = 0.0
    for i in 1:size(X, 2)
        ll += logpdf(d, X[:, i])
    end
    return ll
end


#### Vectorized functions for univariate distributions ####

for fun in [:pdf, :logpdf, :cdf, :logcdf, :ccdf, :logccdf, :invlogcdf, :invlogccdf, :quantile, :cquantile]
    fun! = symbol(string(fun, '!'))

    @eval begin
        function ($fun!)(r::AbstractArray, d::UnivariateDistribution, X::AbstractArray)
            if length(r) != length(X)
                throw(ArgumentError("Inconsistent array dimensions."))
            end
            for i in 1 : length(X)
                r[i] = ($fun)(d, X[i])
            end
            r
        end

        function ($fun)(d::UnivariateDistribution, X::AbstractArray)
            ($fun!)(Array(Float64, size(X)), d, X)
        end
    end
end

#### Vectorized functions for multivariate distributions ####

function logpdf!(r::AbstractArray, d::MultivariateDistribution, x::AbstractMatrix)
    n::Int = size(x, 2)
    if length(r) != n
        throw(ArgumentError("Inconsistent array dimensions."))
    end
    for i in 1 : n
        r[i] = logpdf(d, x[:, i])
    end
    r
end

function logpdf(d::MultivariateDistribution, X::AbstractMatrix)  
    logpdf!(Array(Float64, size(X, 2)), d, X)
end

function pdf!(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix)
    logpdf!(r, d, X)  # size checking is done by logpdf!
    for i in 1 : size(X, 2)
        r[i] = exp(r[i])
    end
    r
end

function pdf(d::MultivariateDistribution, X::AbstractMatrix)
    pdf!(Array(Float64, size(X, 2)), d, X)
end


#### logpmf & pmf for discrete distributions ####

logpmf(d::DiscreteDistribution, args::Any...) = logpdf(d, args...)
logpmf!(r::AbstractArray, d::DiscreteDistribution, args::Any...) = logpdf!(r, d, args...)
pmf(d::DiscreteDistribution, args::Any...) = pdf(d, args...)


#### Sampling: rand & rand! ####

# default: inverse transform sampling
rand(d::UnivariateDistribution) = quantile(d, rand())

function Base.sprand(m::Integer, n::Integer, density::Real, d::Distribution)
    return sprand(m, n, density, n -> rand(d, n))
end


# Fitting

function suffstats{D<:Distribution}(dt::Type{D}, xs...) 
    argtypes = tuple(D, map(typeof, xs)...)
    error("suffstats is not implemented for $argtypes.")
end

fit_mle{D<:UnivariateDistribution}(dt::Type{D}, x::Array) = fit_mle(D, suffstats(D, x))
fit_mle{D<:UnivariateDistribution}(dt::Type{D}, x::Array, w::Array) = fit_mle(D, suffstats(D, x, w))

fit_mle{D<:MultivariateDistribution}(dt::Type{D}, x::Matrix) = fit_mle(D, suffstats(D, x))
fit_mle{D<:MultivariateDistribution}(dt::Type{D}, x::Matrix, w::Array) = fit_mle(D, suffstats(D, x, w))

fit{D<:Distribution}(dt::Type{D}, x) = fit_mle(D, x)
fit{D<:Distribution}(dt::Type{D}, args...) = fit_mle(D, args...)


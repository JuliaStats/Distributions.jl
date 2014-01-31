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

function insupport(d::UnivariateDistribution, X::Array)
    for x in X; insupport(d, x) || return false; end
    true
end

function insupport(t::DataType, X::Array)
    for x in X; insupport(t, x) || return false; end
    true
end

function insupport(d::MultivariateDistribution, X::Matrix)
    for i in 1 : size(X, 2)
        if !insupport(d, X[:, i])  # short-circuit is generally faster
            return false
        end
    end
    return true
end

function insupport(d::MatrixDistribution, X::Array)
    for i in 1 : size(X, 3)
        if !insupport(d, X[:, :, i])
            return false
        end
    end
    return true
end

# generic function to get number of samples

nsamples{D<:UnivariateDistribution}(dt::Type{D}, x::Array) = length(x)
nsamples{D<:MultivariateDistribution}(dt::Type{D}, x::Matrix) = size(x, 2)
nsamples{D<:MatrixDistribution,T}(dt::Type{D}, x::Array{Matrix{T}}) = length(x)

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


### Gradient (derivative of logpdf)

gradloglik(d::UnivariateDistribution, x::Real) = gradloglik(d, float64(x))

#### Sampling: rand & rand! ####

function rand!(d::UnivariateDistribution, A::Array)
    for i in 1:length(A)
        A[i] = rand(d)
    end
    return A
end

function rand(d::ContinuousUnivariateDistribution, dims::Dims)
    return rand!(d, Array(Float64, dims))
end

function rand(d::DiscreteUnivariateDistribution, dims::Dims)
    return rand!(d, Array(Int, dims))
end

function rand(d::UnivariateDistribution, dim1::Integer, dims::Integer...)
    return rand(d, map(int, tuple(dim1,dims...)))
end

function rand(d::ContinuousMultivariateDistribution)
    return rand!(d, Array(Float64, dim(d)))
end

function rand(d::DiscreteMultivariateDistribution)
    return rand!(d, Array(Int, dim(d)))
end

function rand(d::ContinuousMultivariateDistribution, n::Integer)
    return rand!(d, Array(Float64, dim(d), n))
end

function rand(d::DiscreteMultivariateDistribution, n::Integer)
    return rand!(d, Array(Int, dim(d), n))
end

function rand(d::MatrixDistribution, n::Integer)
    return rand!(d, Array(Matrix{Float64}, n))
end

function rand!(d::MultivariateDistribution, X::Matrix)
    if size(X, 1) != dim(d)
        error("Inconsistent argument dimensions")
    end
    for i in 1 : size(X, 2)
        X[:, i] = rand(d)
    end
    X
end

function rand!(d::MatrixDistribution, X::Array{Matrix{Float64}})
    for i in 1:length(X)
        X[i] = rand(d)
    end
    return X
end


function sprand(m::Integer, n::Integer, density::Real, d::Distribution)
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


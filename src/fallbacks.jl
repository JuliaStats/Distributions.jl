##############################################################################
#
# Fallback methods, usually overridden for specific distributions
#
##############################################################################

binaryentropy(d::Distribution) = entropy(d) / log(2)

function cor(d::MultivariateDistribution)
    R = copy(d.cov)
    m, n = size(R)
    for j in 1:n
        for i in 1:n
            R[i, j] = d.cov[i, j] / sqrt(d.cov[i, i] * d.cov[j, j])
        end
    end
    return R
end

cov(d::MultivariateDistribution) = var(d)

ccdf(d::UnivariateDistribution, q::Real) = 1.0 - cdf(d, q)

cquantile(d::UnivariateDistribution, p::Real) = quantile(d, 1.0 - p)

function deviance{M<:Real,Y<:Real,W<:Real}(d::Distribution,
                                           mu::AbstractArray{M},
                                           y::AbstractArray{Y},
                                           wt::AbstractArray{W})
    promote_shape(size(mu), promote_shape(size(y), size(wt)))
    ans = 0.0
    for i in 1:length(y)
        ans += wt[i] * logpdf(d, mu[i], y[i])
    end
    return -2.0 * ans
end

function devresid(d::Distribution, y::Real, mu::Real, wt::Real)
    return -2.0 * wt * logpdf(d, mu, y)
end

function devresid{Y<:Real,M<:Real,W<:Real}(d::Distribution,
                                           y::AbstractArray{Y},
                                           mu::AbstractArray{M},
                                           wt::AbstractArray{W})
    R = Array(Float64, promote_shape(size(y), promote_shape(size(mu), size(wt))))
    for i in 1:length(mu)
        R[i] = devresid(d, y[i], mu[i], wt[i])
    end
    return R
end

function devresid(d::Distribution, y::Vector{Float64},
                  mu::Vector{Float64}, wt::Vector{Float64})
    return [devresid(d, y[i], mu[i], wt[i]) for i in 1:length(y)]
end

function insupport{T <: Real}(d::UnivariateDistribution,
                              X::Array{T})
    for el in X
        if !insupport(d, el)
            return false
        end
    end
    return true
end

function insupport{T <: Real}(d::MultivariateDistribution,
                              X::Matrix{T})
    res = true
    for i in 1:size(X, 2)
        res &= insupport(d, X[:, i])
    end
    return res
    # TODO: Determine which is faster
    # for i in 1:size(X, 2)
    #     if !insupport(d, X[:, i])
    #         return false
    #     end
    # end
    # return true
end

function insupport{T <: Real}(d::MatrixDistribution,
                              X::Array{T})
    res = true
    for i in 1:size(X, 3)
        res &= insupport(d, X[:, :, i])
    end
    return res
end

insupport(d::Distribution, x::Any) = false

invlogccdf(d::Distribution, lp::Real) = quantile(d, -expm1(lp))

invlogcdf(d::Distribution, lp::Real) = quantile(d, exp(lp))

isplatykurtic(d::Distribution) = kurtosis(d) > 0.0

isleptokurtic(d::Distribution) = kurtosis(d) < 0.0

ismesokurtic(d::Distribution) = kurtosis(d) == 0.0

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

logcdf(d::Distribution, q::Real) = log(cdf(d,q))

logccdf(d::Distribution, q::Real) = log(ccdf(d,q))

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

logpdf(d::Distribution, x::Real) = log(pdf(d,x))

function logpdf!(r::AbstractArray, d::UnivariateDistribution, x::AbstractArray)
    if size(x) != size(r)
        throw(ArgumentError("Inconsistent array dimensions."))
    end    
    for i in 1:length(x)
        r[i] = logpdf(d, x[i])
    end
end

function logpdf(d::MultivariateDistribution, x::AbstractMatrix)
    n::Int = size(x, 2)
    r = Array(Float64, n)   
    for i in 1:n
        r[i] = logpdf(d, x[:, i])
    end
    r
end

# function logpdf{T <: Real}(d::Dirichlet, x::Matrix{T})
#     r = Array(Float64, size(x, 2))
#     logpdf!(r, d, x)
#     return r
# end

function logpdf!(r::AbstractArray, d::MultivariateDistribution, x::AbstractMatrix)
    n::Int = size(x, 2)
    if length(r) != n
        throw(ArgumentError("Inconsistent array dimensions."))
    end
    for i = 1:n
        r[i] = logpdf(d, x[:, i])
    end
end

logpmf(d::DiscreteDistribution, args::Any...) = logpdf(d, args...)

function logpmf!(r::AbstractArray, d::DiscreteDistribution, args::Any...)
    return logpdf!(r, d, args...)
end

function mustart{Y<:Real,W<:Real}(d::Distribution,
                                  y::AbstractArray{Y},
                                  wt::AbstractArray{W})
    M = Array(Float64, promote_shape(size(y), size(wt)))
    for i in 1:length(M)
        M[i] = mustart(d, y[i], wt[i])
    end
    return M
end

pmf(d::DiscreteDistribution, args::Any...) = pdf(d, args...)

function pdf(d::MultivariateDistribution, X::AbstractMatrix)
    r = Array(Float64, size(X, 2))
    return pdf!(r, d, X)
end

function pdf!(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix)
    for i in 1:size(X, 2)
        r[i] = pdf(d, X[:, i])
    end
    return r
end

# Activate after checking all distributions have something written out
# median(d::UnivariateDistribution) = quantile(d, 0.5)

function rand!(d::UnivariateDistribution, A::Array)
    for i in 1:length(A)
        A[i] = rand(d)
    end
    return A
end

function rand(d::ContinuousDistribution, dims::Dims)
    return rand!(d, Array(Float64, dims))
end

function rand(d::DiscreteDistribution, dims::Dims)
    return rand!(d, Array(Int, dims))
end

function rand(d::NonMatrixDistribution, dims::Integer...)
    return rand(d, map(int, dims))
end

function rand(d::ContinuousMultivariateDistribution, dims::Integer)
    return rand!(d, Array(Float64, length(mean(d)), dims))
end

function rand(d::DiscreteMultivariateDistribution, dims::Integer)
    return rand!(d, Array(Int, length(mean(d)), dims))
end

function rand(d::MatrixDistribution, dims::Integer)
    return rand!(d, Array(Matrix{Float64}, int(dims)))
end

function rand!(d::MultivariateDistribution, X::Matrix)
    k = length(mean(d))
    m, n = size(X)
    if m != k
        error("Wrong dimensions")
    end
    for i in 1:n
        X[:, i] = rand(d)
    end
    return X
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

std(d::Distribution) = sqrt(var(d))

function var{M <: Real}(d::Distribution, mu::AbstractArray{M})
    V = similar(mu, Float64)
    for i in 1:length(mu)
        V[i] = var(d, mu[i])
    end
    return V
end

# Vectorize methods
for f in (:pdf, :logpdf, :cdf, :logcdf, :ccdf, :logccdf, :quantile,
          :cquantile, :invlogcdf, :invlogccdf)
    @eval begin  
        function ($f)(d::UnivariateDistribution, x::AbstractArray)
            res = Array(Float64, size(x))
            for i in 1:length(res)
                res[i] = ($f)(d, x[i])
            end
            return res
        end
    end
end

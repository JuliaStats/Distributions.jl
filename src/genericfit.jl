# generic functions for distribution fitting

function suffstats{D<:Distribution}(dt::Type{D}, xs...)
    argtypes = tuple(D, map(typeof, xs)...)
    error("suffstats is not implemented for $argtypes.")
end

"""
    fit_mle(D, x)

Fit a distribution of type `D` to a given data set `x`.

- For univariate distribution, x can be an array of arbitrary size.
- For multivariate distribution, x should be a matrix, where each column is a sample.
"""
fit_mle(D, x)

"""
    fit_mle(D, x, w)

Fit a distribution of type `D` to a weighted data set `x`, with weights given by `w`.

Here, `w` should be an array with length `n`, where `n` is the number of samples contained in `x`.
"""
fit_mle(D, x, w)

fit_mle{D<:UnivariateDistribution}(dt::Type{D}, x::AbstractArray) = fit_mle(D, suffstats(D, x))
fit_mle{D<:UnivariateDistribution}(dt::Type{D}, x::AbstractArray, w::AbstractArray) = fit_mle(D, suffstats(D, x, w))

fit_mle{D<:MultivariateDistribution}(dt::Type{D}, x::AbstractMatrix) = fit_mle(D, suffstats(D, x))
fit_mle{D<:MultivariateDistribution}(dt::Type{D}, x::AbstractMatrix, w::AbstractArray) = fit_mle(D, suffstats(D, x, w))

fit{D<:Distribution}(dt::Type{D}, x) = fit_mle(D, x)
fit{D<:Distribution}(dt::Type{D}, args...) = fit_mle(D, args...)

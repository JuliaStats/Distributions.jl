# generic functions for distribution fitting

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

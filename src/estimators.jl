# uniform interface for model estimation

export Estimator, MLEstimator
export nsamples, estimate

abstract type Estimator{D<:Distribution} end

nsamples(e::Estimator{D}, x::Array) where {D<:UnivariateDistribution} = length(x)
nsamples(e::Estimator{D}, x::Matrix) where {D<:MultivariateDistribution} = size(x, 2)

mutable struct MLEstimator{D<:Distribution} <: Estimator{D} end
MLEstimator(::Type{D}) where {D<:Distribution} = MLEstimator{D}()

estimate(e::MLEstimator{D}, x) where {D<:Distribution} = fit_mle(D, x)
estimate(e::MLEstimator{D}, x, w) where {D<:Distribution} = fit_mle(D, x, w)

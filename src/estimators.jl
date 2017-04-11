# uniform interface for model estimation

export Estimator, MLEstimator
export nsamples, estimate

@compat abstract type Estimator{D<:Distribution} end

nsamples{D<:UnivariateDistribution}(e::Estimator{D}, x::Array) = length(x)
nsamples{D<:MultivariateDistribution}(e::Estimator{D}, x::Matrix) = size(x, 2)

type MLEstimator{D<:Distribution} <: Estimator{D} end
MLEstimator{D<:Distribution}(::Type{D}) = MLEstimator{D}()

estimate{D<:Distribution}(e::MLEstimator{D}, x) = fit_mle(D, x)
estimate{D<:Distribution}(e::MLEstimator{D}, x, w) = fit_mle(D, x, w)

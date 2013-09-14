# uniform interface for model estimation

export Estimator, MLEstimator, MAPEstimator
export nsamples, estimate, prior_score

abstract Estimator{D<:Distribution}

nsamples{D<:UnivariateDistribution}(e::Estimator{D}, x::Array) = length(x)
nsamples{D<:MultivariateDistribution}(e::Estimator{D}, x::Matrix) = size(x, 2)

type MLEstimator{D<:Distribution} <: Estimator{D} end
MLEstimator{D<:Distribution}(::Type{D}) = MLEstimator{D}()

estimate{D<:Distribution}(e::MLEstimator{D}, x) = fit_mle(D, x)
estimate{D<:Distribution}(e::MLEstimator{D}, x, w) = fit_mle(D, x, w)

prior_score{D<:Distribution}(e::MLEstimator{D}, d::D) = 0.

immutable MAPEstimator{D<:Distribution,Pri} <: Estimator{D} 
	pri::Pri
end
MAPEstimator{D<:Distribution,Pri}(::Type{D}, pri::Pri) = MAPEstimator{D,Pri}(pri)

estimate{D<:Distribution,Pri}(e::MAPEstimator{D,Pri}, x) = fit_map(e.pri, D, x)
estimate{D<:Distribution,Pri}(e::MAPEstimator{D,Pri}, x, w) = fit_map(e.pri, D, x, w) 

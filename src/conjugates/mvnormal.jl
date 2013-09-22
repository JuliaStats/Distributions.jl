# Conjugates for Multivariate Normal


##### 
#
# Cases where covariance is known, the goal is to infer the posterior of mu
#

function posterior(prior::GenericMultivariateNormal, ss::GenericMvNormalKnownSigmaStats)
    invΣ0 = inv(prior.Σ)
    μ0 = prior.μ
    invΣp = add_scal(invΣ0, ss.invΣ, ss.tw)
    Σp = inv(invΣp)
    μp = Σp * add!(invΣ0 * μ0, ss.invΣ * ss.sx)
	return gmvnormal(μp, Σp)
end



function posterior{T<:Real}(prior::(MultivariateNormal, Matrix{Float64}), ::Type{MultivariateNormal}, X::Matrix{T}) 
	pri_μ::MultivariateNormal, Σ::Matrix{Float64} = prior
	posterior(pri_μ, suffstats(MvNormalKnownSigma(Σ), X))
end

function posterior{T<:Real}(prior::(MultivariateNormal, Matrix{Float64}), ::Type{MultivariateNormal}, X::Matrix{T}, w::Array{Float64}) 
	pri_μ::MultivariateNormal, Σ::Matrix{Float64} = prior
	posterior(pri_μ, suffstats(MvNormalKnownSigma(Σ), X, w))
end

function posterior_sample{T<:Real}(prior::(MultivariateNormal, Matrix{Float64}), ::Type{MultivariateNormal}, X::Matrix{T})
    mu = rand(posterior(prior[1], suffstats(MvNormalKnownSigma(prior[2]), X)))
    return MultivariateNormal(mu, prior[2]) 
end

function posterior_sample{T<:Real}(prior::(MultivariateNormal, Matrix{Float64}), ::Type{MultivariateNormal}, X::Matrix{T}, w::Array{Float64})
    mu = rand(posterior(prior[1], suffstats(MvNormalKnownSigma(prior[2]), X, w)))
    return MultivariateNormal(mu, prior[2]) 
end


# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

import NumericExtensions.sum

immutable MvNormalKnownSigma <: GenerativeFormulation
    Sigma::Matrix{Float64}

    function MvNormalKnownSigma(S::Matrix{Float64})
        # TODO: Error checking, maybe
        new(S)
    end
end

immutable MvNormalKnownSigmaStats
    Sigma::Matrix{Float64}      # known covariance
    s::Vector{Float64}          # (weighted) sum of x
    tw::Float64                 # total sample weight

    function MvNormalKnownSigmaStats(S::Matrix{Float64}, 
                                     s::Vector{Float64},
                                     tw::Float64)
        new(S, s, float64(tw))
    end
end

function suffstats{T<:Real}(g::MvNormalKnownSigma, X::Matrix{T})
    d, n = size(X)

    s = X[:,1]
    for j in 2:n
        for i in 1:d
            @inbounds s[i] += X[i,j]
        end
    end

    MvNormalKnownSigmaStats(g.Sigma, s, float64(n))    
end

function suffstats{T<:Real}(g::MvNormalKnownSigma, X::Matrix{T}, w::Array{T})
    d, n = size(X)

    # Could use NumericExtensions or BLAS
    tw = w[1]
    s = w[1] .* X[:,1]
    for j in 2:n
        @inbounds wj = w[j]
        for i in 1:d
            @inbounds s[i] += wj * X[i,j]
        end
        tw += wj
    end

    MvNormalKnownSigmaStats(g.Sigma, s, tw)    
end

function posterior(prior::MultivariateNormal, ss::MvNormalKnownSigmaStats)
    Sigma0inv = inv(prior.Σ)
    mu0 = prior.μ
    Sigmainv = inv(ss.Sigma)

    SigmaN = inv(Sigma0inv + ss.tw*Sigmainv)
    muN = SigmaN*(Sigmainv*ss.s + Sigma0inv*mu0)

	return MultivariateNormal(muN, SigmaN)	
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


### Beta -- Bernoulli or Binomial

posterior(prior::Beta, ss::BernoulliStats) = Beta(prior.alpha + ss.cnt1, prior.beta + ss.cnt0)

posterior(prior::Beta, ss::BinomialStats) = Beta(prior.alpha + ss.ns, prior.beta + (ss.ne * ss.n - ss.ns))

function posterior{T<:Real}(prior::Beta, ::Type{Binomial}, n::Integer, x::Array{T})
	posterior(prior, suffstats(Binomial, n, x))
end

function posterior{T<:Real}(prior::Beta, ::Type{Binomial}, n::Integer, x::Array{T}, w::Array{Float64})
	posterior(prior, suffstats(Binomial, n, x, w))
end

### Dirichlet -- Categorical or Multinomial

function posterior{T<:Real}(prior::Dirichlet, ::Type{Categorical}, x::Array{T})
	α = copy(prior.alpha)
	for i in 1:length(x)
		@inbounds xi = x[i]
		α[xi] += 1.0   # cannot put @inbounds here, as no guarantee xi is inbound
	end
	return Dirichlet(α)
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Categorical}, x::Array{T}, w::Array{Float64})
	if length(x) != length(w)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	α = copy(prior.alpha)
	for i in 1:length(x)
		@inbounds xi = x[i]
		@inbounds wi = w[i]
		α[xi] += wi
	end
	return Dirichlet(α)
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, x::Vector{T})
	return Dirichlet(prior.alpha + x)
end

# each column are counts for one experiment
function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T})
	d::Int = dim(prior)
	if d != size(X, 1)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	n = size(X, 2)
	α = copy(prior.alpha)
	o = 0
	for j in 1:n
		for i in 1:d
			@inbounds α[i] += X[o+i]
		end
		o += d
	end
	return Dirichlet(α)
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T}, w::Array{Float64})
	d::Int = dim(prior)
	if d != size(X, 1)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	n = size(X, 2)
	if n != length(w)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	α = copy(prior.alpha)
	o = 0
	for j in 1:n
		@inbounds wj = w[j]
		for i in 1:d
			@inbounds α[i] += X[o+i] * wj
		end
		o += d
	end
	return Dirichlet(α)
end


### Gamma -- Exponential

function posterior(prior::Gamma, ss::ExponentialStats)
	return Gamma(prior.shape + ss.sw, 1.0 / (rate(prior) + ss.sx))
end



### For Normal distributions

function posterior(prior::Normal, ss::NormalKnownSigmaStats)
	μ0 = prior.μ
	c0 = 1.0 / abs2(prior.σ)
	c1 = 1.0 / abs2(ss.σ)

	τ = c0 + ss.tw * c1
	μ1 = (μ0 * c0 + ss.s * c1) / τ
	σ1 = sqrt(1 / τ)
	return Normal(μ1, σ1)	
end

function posterior(prior::InvertedGamma, ss::NormalKnownMuStats)
	α1 = prior.shape + ss.tw / 2
	β1 = rate(prior) + ss.s2 / 2
	return InvertedGamma(α1, 1.0 / β1)
end

function posterior{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}) 
	pri_μ::Normal = prior[1]
	σ::Float64 = prior[2]
	posterior(pri_μ, suffstats(NormalKnownSigma(σ), x))
end

function posterior{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	pri_μ::Normal = prior[1]
	σ::Float64 = prior[2]
	posterior(pri_μ, suffstats(NormalKnownSigma(σ), x, w))
end

function posterior{T<:Real}(prior::(Float64, InvertedGamma), ::Type{Normal}, x::Array{T}) 
	μ::Float64 = prior[1]
	pri_σ::InvertedGamma = prior[2]
	posterior(pri_σ, suffstats(NormalKnownMu(μ), x))
end

function posterior{T<:Real}(prior::(Float64, InvertedGamma), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	μ::Float64 = prior[1]
	pri_σ::InvertedGamma = prior[2]
	posterior(pri_σ, suffstats(NormalKnownMu(μ), x, w))
end


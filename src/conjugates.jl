
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

function dirichlet_posupdate!{T<:Integer}(α::Vector{Float64}, ::Type{Categorical}, x::Array{T})
	for i in 1:length(x)
		@inbounds xi = x[i]
		α[xi] += 1.0   # cannot put @inbounds here, as no guarantee xi is inbound
	end
	return α
end

function dirichlet_posupdate!{T<:Integer}(α::Vector{Float64}, ::Type{Categorical}, x::Array{T}, w::Array{Float64})
	if length(x) != length(w)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	for i in 1:length(x)
		@inbounds xi = x[i]
		@inbounds wi = w[i]
		α[xi] += wi
	end
	return α
end

# each column are counts for one experiment
function dirichlet_posupdate!{T<:Real}(α::Vector{Float64}, ::Type{Multinomial}, X::Matrix{T})
	d::Int = length(α)
	if d != size(X, 1)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	n = size(X, 2)
	o = 0
	for j = 1:n
		for i = 1:d
			@inbounds α[i] += X[o+i]
		end
		o += d
	end
	return α
end

function dirichlet_posupdate!{T<:Real}(α::Vector{Float64}, ::Type{Multinomial}, X::Matrix{T}, w::Array{Float64})
	d::Int = length(α)
	if d != size(X, 1)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	n = size(X, 2)
	if n != length(w)
		throw(ArgumentError("Inconsistent argument dimensions."))
	end
	o = 0
	for j = 1:n
		@inbounds wj = w[j]
		for i = 1:d
			@inbounds α[i] += X[o+i] * wj
		end
		o += d
	end
	return α
end

function posterior{T<:Integer}(prior::Dirichlet, ::Type{Categorical}, x::Array{T})
	Dirichlet(dirichlet_posupdate!(copy(prior.alpha), Categorical, x))
end

function posterior{T<:Integer}(prior::Dirichlet, ::Type{Categorical}, x::Array{T}, w::Array{Float64})
	Dirichlet(dirichlet_posupdate!(copy(prior.alpha), Categorical, x, w))
end

function posterior_mode{T<:Integer}(prior::Dirichlet, ::Type{Categorical}, x::Array{T})
	α = dirichlet_posupdate!(copy(prior.alpha), Categorical, x)
	dirichlet_mode!(α, α, sum(α))
end

function posterior_mode{T<:Integer}(prior::Dirichlet, ::Type{Categorical}, x::Array{T}, w::Array{Float64})
	α = dirichlet_posupdate!(copy(prior.alpha), Categorical, x, w)
	dirichlet_mode!(α, α, sum(α))
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, x::Vector{T})
	return Dirichlet(prior.alpha + x)
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T})
	Dirichlet(dirichlet_posupdate!(copy(prior.alpha), Multinomial, X))
end

function posterior{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T}, w::Array{Float64})
	Dirichlet(dirichlet_posupdate!(copy(prior.alpha), Multinomial, X, w))
end

function posterior_mode{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, x::Vector{T})
	α = prior.alpha + x
	dirichlet_mode!(α, α, sum(α))
end

function posterior_mode{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T})
	α = dirichlet_posupdate!(copy(prior.alpha), Multinomial, X)
	dirichlet_mode!(α, α, sum(α))
end

function posterior_mode{T<:Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T}, w::Array{Float64})
	α = dirichlet_posupdate!(copy(prior.alpha), Multinomial, X, w)
	dirichlet_mode!(α, α, sum(α))
end



### Gamma -- Exponential (rate)

function posterior(prior::Gamma, ss::ExponentialStats)
	return Gamma(prior.shape + ss.sw, 1.0 / (rate(prior) + ss.sx))
end

posterior_make(::Type{Exponential}, θ::Float64) = Exponential(1.0 / θ)


### For Normal distributions

# known sigma (prior on mu)

function posterior(prior::Normal, ss::NormalKnownSigmaStats)
	μ0 = prior.μ
	c0 = 1.0 / abs2(prior.σ)
	c1 = 1.0 / abs2(ss.σ)

	τ = c0 + ss.tw * c1
	μ1 = (μ0 * c0 + ss.s * c1) / τ
	σ1 = sqrt(1 / τ)
	return Normal(μ1, σ1)	
end

function posterior{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}) 
	pri_μ::Normal, σ::Float64 = prior
	posterior(pri_μ, suffstats(NormalKnownSigma(σ), x))
end

function posterior{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	pri_μ::Normal, σ::Float64 = prior
	posterior(pri_μ, suffstats(NormalKnownSigma(σ), x, w))
end

function posterior_mode(prior::Normal, ss::NormalKnownSigmaStats)
	μ0 = prior.μ
	c0 = 1.0 / abs2(prior.σ)
	c1 = 1.0 / abs2(ss.σ)
	return (μ0 * c0 + ss.s * c1) / (c0 + ss.tw * c1)
end

function posterior_mode{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}) 
	pri_μ::Normal, σ::Float64 = prior
	posterior_mode(pri_μ, suffstats(NormalKnownSigma(σ), x))
end

function posterior_mode{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	pri_μ::Normal, σ::Float64 = prior
	posterior_mode(pri_μ, suffstats(NormalKnownSigma(σ), x, w))
end

function fit_map{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T})
	pri_μ::Normal, σ::Float64 = prior
	μ = posterior_mode(pri_μ, suffstats(NormalKnownSigma(σ), x))
	Normal(μ, σ)
end

function fit_map{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}, w::Array{Float64})
	pri_μ::Normal, σ::Float64 = prior
	μ = posterior_mode(pri_μ, suffstats(NormalKnownSigma(σ), x, w))
	Normal(μ, σ)
end


# known mu (prior on sigma)

function posterior(prior::InverseGamma, ss::NormalKnownMuStats)
	α1 = prior.shape + ss.tw / 2
	β1 = prior.scale + ss.s2 / 2
	return InverseGamma(α1, β1)
end

function posterior{T<:Real}(prior::(Float64, InverseGamma), ::Type{Normal}, x::Array{T}) 
	μ::Float64 = prior[1]
	pri_σ::InverseGamma = prior[2]
	posterior(pri_σ, suffstats(NormalKnownMu(μ), x))
end

function posterior{T<:Real}(prior::(Float64, InverseGamma), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	μ::Float64 = prior[1]
	pri_σ::InverseGamma = prior[2]
	posterior(pri_σ, suffstats(NormalKnownMu(μ), x, w))
end


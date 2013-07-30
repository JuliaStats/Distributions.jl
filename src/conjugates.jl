
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


### TODO: explore ways to deal with Normal

# Normal(μ0, σ0) prior on μ
# Known generative standard deviation σ
# Normal likelihood
function posterior{T <: Real}(prior::Normal,
	                          σ::Real,
	                          ::Type{Normal},
	                          x::Vector{T})
	n = length(x)
	μ0, σ0 = mean(prior), std(prior)
	τ = 1 / σ0^2 + n / σ^2
	μ1 = (μ0 / σ0^2 + sum(x) / σ^2) / τ
	σ1 = sqrt(1 / τ)
	return Normal(μ1, σ1)
end

# Known generative variance μ
# InvertedGamma(α, β) prior on σ
# Normal likelihood
function posterior{T <: Real}(μ::Real,
	                          prior::InvertedGamma,
	                          ::Type{Normal},
	                          x::Vector{T})
	n = length(x)
	α0, β0 = prior.shape, 1 / prior.scale
	sqsum = 0.0
	for i in 1:n
		sqsum += (x[i] - μ)^2
	end
	α1 = α0 + n / 2
	β1 = β0 + sqsum / 2
	return InvertedGamma(α1, 1 / β1)
end

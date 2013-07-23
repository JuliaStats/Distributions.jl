# Bernoulli likelihood
# Beta prior
# x contains 0-1 data
function posterior{T <: Real}(prior::Beta, ::Type{Bernoulli}, x::Vector{T})
	a, n = sum(x), length(x)
	b = n - a
	return Beta(prior.alpha + a, prior.beta + b)
end

# Binomial likelihood
# Beta prior
# X is a matrix with a column of successes and a column of trials
function posterior{T <: Real}(prior::Beta, ::Type{Binomial}, X::Matrix{T})
	a, n = sum(X, 1)
	b = n - a
	return Beta(prior.alpha + a, prior.beta + b)
end

# Categorical likelihood
# Dirichlet prior
# x contains 1-k data
function posterior{T <: Real}(prior::Dirichlet, ::Type{Categorical}, x::Vector{T})
	alpha = copy(prior.alpha)
	for i in 1:length(x)
		alpha[x[i]] += 1.0
	end
	return Dirichlet(alpha)
end

# Multinomial likelihood
# Dirichlet prior
# x is a single Multinomial draw
function posterior{T <: Real}(prior::Dirichlet, ::Type{Multinomial}, x::Vector{T})
	return Dirichlet(prior.alpha + x)
end

# Multinomial likelihood
# Dirichlet prior
# X is a matrix with Multinomial draws as columns
function posterior{T <: Real}(prior::Dirichlet, ::Type{Multinomial}, X::Matrix{T})
	p, n = size(X)
	u = zeros(p)
	for j in 1:n
		for i in 1:p
			u[i] += X[i, j]
		end
	end
	return Dirichlet(prior.alpha + u)
end

# Gamma(α, β) prior on scale
# Exponential likelihood
function posterior{T <: Real}(prior::Gamma,
	                          ::Type{Exponential},
	                          x::Vector{T})
	n = length(x)
	α, β = prior.shape, 1 / prior.scale
	return Gamma(α + n, 1 / (β + sum(x)))
end

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

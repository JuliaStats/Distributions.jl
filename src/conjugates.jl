# Bernoulli likelihood
# Beta prior
# x contains 0-1 data
function posterior(::Type{Bernoulli}, prior::Beta, x::Vector)
	a, n = sum(x), length(x)
	b = n - a
	return Beta(prior.alpha + a, prior.beta + b)
end

# Binomial likelihood
# Beta prior
# X is a matrix with a column of successes and a column of trials
function posterior(::Type{Binomial}, prior::Beta, X::Matrix)
	a, n = sum(X, 1)
	b = n - a
	return Beta(prior.alpha + a, prior.beta + b)
end

# Categorical likelihood
# Dirichlet prior
# x contains 1-k data
function posterior(::Type{Categorical}, prior::Dirichlet, x::Vector)
	alpha = copy(prior.alpha)
	for i in 1:length(x)
		alpha[x[i]] += 1.0
	end
	return Dirichlet(alpha)
end

# Multinomial likelihood
# Dirichlet prior
# x is a single Multinomial draw
function posterior(::Type{Multinomial}, prior::Dirichlet, x::Vector)
	return Dirichlet(prior.alpha + x)
end

# Multinomial likelihood
# Dirichlet prior
# X is a matrix with Multinomial draws as columns
function posterior(::Type{Multinomial}, prior::Dirichlet, X::Matrix)
	p, n = size(X)
	u = zeros(p)
	for j in 1:n
		for i in 1:p
			u[i] += X[i, j]
		end
	end
	return Dirichlet(prior.alpha + u)
end

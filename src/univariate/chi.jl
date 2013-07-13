immutable Chi <: ContinuousUnivariateDistribution
    df::Float64

    Chi(df::Real) = new(float64(df))
end

cdf(d::Chi, x::Real) = regularized_gamma(d.df / 2.0, x^2 / 2.0)

function mean(d::Chi)
	return sqrt(2.0) * gamma((d.df + 1.0) / 2.0) / gamma(d.df / 2.0)
end

function modes(d::Chi)
	if d.df < 1.0
		error("Modes undefined for k < 1")
	else
		return [sqrt(d.df - 1)]
	end
end

var(d::Chi) = d.df - mean(d)^2

function skewness(d::Chi)
	μ, σ = mean(d), std(d)
	(μ / σ^2) * (1.0 - 2.0 * σ^2)
end

function kurtosis(d::Chi)
	μ, σ, γ = mean(d), std(d), skewness(d)
	(2.0 / σ^2) * (1 - μ * σ * γ - σ^2)
end

function entropy(d::Chi)
	lgamma(k / 2.0) + 0.5 * (k - log(2.0) - (k - 1.0) * digamma(k / 2.0))
end

function pdf(d::Chi, x::Real)
	k = d.df
	return (2.0^(1.0 - k / 2.0) * x^(k - 1.0) * exp(-x^2 / 2.0)) /
	       gamma(k / 2.0)
end

function entropy(d::Chi)
	k = d.df
	return lgamma(k / 2.0) - log(sqrt(2.0)) -
	       ((k - 1.0) / 2.0) * digamma(k / 2.0) + k / 2.0
end

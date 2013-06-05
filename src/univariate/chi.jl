immutable Chi <: ContinuousUnivariateDistribution
    df::Float64
end

function entropy(d::Chi)
	k = d.df
	return lgamma(k / 2.0) - log(sqrt(2.0)) -
	       ((k - 1.0) / 2.0) * digamma(k / 2.0) + k / 2.0
end

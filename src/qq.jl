immutable QQPair
	qx::Vector{Float64}
	qy::Vector{Float64}
end

function qqbuild(x::Vector, y::Vector)
	n = min(length(x), length(y))
	grid = [0.0:(1 / (n - 1)):1.0;]
	qx = quantile(x, grid)
	qy = quantile(y, grid)
	return QQPair(qx, qy)
end

function qqbuild(x::Vector, d::UnivariateDistribution)
	stp = inv(length(x))
	return QQPair(quantile(d, stp/2 : step : one(step)), sort(x))
end

qqbuild(d::UnivariateDistribution, x::Vector) = qqbuild(x, d)

immutable Poisson <: DiscreteUnivariateDistribution
    lambda::Float64
    function Poisson(l::Real)
    	if l > 0.0
    		new(float64(l))
    	else
    		error("lambda must be positive")
    	end
    end
end

Poisson() = Poisson(1.0)

@_jl_dist_1p Poisson pois

insupport(d::Poisson, x::Number) = isinteger(x) && 0.0 <= x

function logpdf(d::Poisson, mu::Real, y::Real)
	return ccall((:dpois, Rmath),
		         Float64,
		         (Float64, Float64, Int32),
		         y, mu, 1)
end

mean(d::Poisson) = d.lambda

var(d::Poisson) = d.lambda

# GLM Methods

function devresid(d::Poisson, y::Real, mu::Real, wt::Real)
	return 2.0 * wt * (xlogxdmu(y, mu) - (y - mu))
end

function devresid(d::Poisson, y::Vector{Float64},
	              mu::Vector{Float64}, wt::Vector{Float64})
    [2.0 * wt[i] * (xlogxdmu(y[i], mu[i]) - (y[i] - mu[i])) for i in 1:length(y)]
end

mustart(d::Poisson, y::Real, wt::Real) = y + 0.1

var(d::Poisson, mu::Real) = mu

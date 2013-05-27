immutable Normal <: ContinuousUnivariateDistribution
    mean::Float64
    std::Float64
    function Normal(mu::Real, sd::Real)
    	if sd > 0.0
    		new(float64(mu), float64(sd))
    	else
    		error("std must be positive")
    	end
    end
end

Normal(mu::Real) = Normal(mu, 1.0)
Normal() = Normal(0.0, 1.0)

const Gaussian = Normal

@_jl_dist_2p Normal norm

entropy(d::Normal) = (1.0 / 2.0) * log(2.0 * pi * e * d.std^2)

insupport(d::Normal, x::Number) = isreal(x) && isfinite(x)

kurtosis(d::Normal) = 0.0

mean(d::Normal) = d.mean

median(d::Normal) = d.mean

modes(d::Normal) = [d.mean]

rand(d::Normal) = d.mean + d.std * randn()

skewness(d::Normal) = 0.0

std(d::Normal) = d.std

var(d::Normal) = d.std^2

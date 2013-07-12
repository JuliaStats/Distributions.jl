immutable Uniform <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64
    function Uniform(a::Real, b::Real)
	    if a < b
	    	new(float64(a), float64(b))
	    else
	    	error("a < b required for range [a, b]")
	    end
	end
end

Uniform() = Uniform(0.0, 1.0)

@_jl_dist_2p Uniform unif

entropy(d::Uniform) = log(d.b - d.a)

insupport(d::Uniform, x::Number) = isreal(x) && d.a <= x <= d.b

kurtosis(d::Uniform) = -6.0 / 5.0

mean(d::Uniform) = (d.a + d.b) / 2.0

median(d::Uniform) = (d.a + d.b) / 2.0

function mgf(d::Uniform, t::Real)
	a, b = d.a, d.b
	return (exp(t * b) - exp(t * a)) / (t * (b - a))
end

function cf(d::Uniform, t::Real)
	a, b = d.a, d.b
	return (exp(im * t * b) - exp(im * t * a)) / (im * t * (b - a))
end

modes(d::Uniform) = error("The uniform distribution has no modes")

rand(d::Uniform) = d.a + (d.b - d.a) * rand()

skewness(d::Uniform) = 0.0

function var(d::Uniform)
	w = d.b - d.a
	return w * w / 12.0
end

function fit_mle{T <: Real}(::Type{Uniform}, x::Vector{T})
	Uniform(min(x), max(x))
end

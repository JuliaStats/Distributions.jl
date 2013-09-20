immutable Uniform <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64
    function Uniform(a::Real, b::Real)
	a < b || error("a < b required for range [a, b]")
	new(float64(a), float64(b))
    end
    Uniform() = new(0.0, 1.0)
end

@_jl_dist_2p Uniform unif

isupperbounded(::Union(Uniform, Type{Uniform})) = true
islowerbounded(::Union(Uniform, Type{Uniform})) = true
isbounded(::Union(Uniform, Type{Uniform})) = true

min(d::Uniform) = d.a
max(d::Uniform) = d.b
insupport(d::Uniform, x::Real) = d.a <= x <= d.b

entropy(d::Uniform) = log(d.b - d.a)

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

mode(d::Uniform) = d.a
modes(d::Uniform) = error("The uniform distribution has no modes")

rand(d::Uniform) = d.a + (d.b - d.a) * rand()

skewness(d::Uniform) = 0.0

function var(d::Uniform)
	w = d.b - d.a
	return w * w / 12.0
end

# fit model

function fit_mle{T <: Real}(::Type{Uniform}, x::Vector{T})
    if isempty(x)
        throw(ArgumentError("x cannot be empty."))
    end

    xmin = xmax = x[1]
    for i = 2:length(x)
        xi = x[i]
        if xi < xmin
            xmin = xi
        elseif xi > xmax
            xmax = xi
        end
    end

    Uniform(xmin, xmax)
end

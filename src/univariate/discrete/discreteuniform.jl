"""
    DiscreteUniform(a,b)

A *Discrete uniform distribution* is a uniform distribution over a consecutive sequence of integers between `a` and `b`, inclusive.

```math
P(X = k) = 1 / (b - a + 1) \\quad \\text{for } k = a, a+1, \\ldots, b.
```

```julia
DiscreteUniform(a, b)   # a uniform distribution over {a, a+1, ..., b}

params(d)       # Get the parameters, i.e. (a, b)
span(d)         # Get the span of the support, i.e. (b - a + 1)
probval(d)      # Get the probability value, i.e. 1 / (b - a + 1)
minimum(d)      # Return a
maximum(d)      # Return b
```

External links

* [Discrete uniform distribution on Wikipedia](http://en.wikipedia.org/wiki/Uniform_distribution_(discrete))
"""
struct DiscreteUniform <: DiscreteUnivariateDistribution
    a::Int
    b::Int
    pv::Float64 # individual probabilities

    function DiscreteUniform(a::Real, b::Real)
        @check_args(DiscreteUniform, a <= b)
        new(a, b, 1.0 / (b - a + 1))
    end
    DiscreteUniform(b::Real) = DiscreteUniform(0, b)
    DiscreteUniform() = new(0, 1, 0.5)
end

@distr_support DiscreteUniform d.a d.b

### Parameters

span(d::DiscreteUniform) = d.b - d.a + 1
probval(d::DiscreteUniform) = d.pv
params(d::DiscreteUniform) = (d.a, d.b)

### Show

show(io::IO, d::DiscreteUniform) = show(io, d, (:a, :b))


### Statistics

mean(d::DiscreteUniform) = middle(d.a, d.b)

median(d::DiscreteUniform) = middle(d.a, d.b)

var(d::DiscreteUniform) = (span(d)^2 - 1.0) / 12.0

skewness(d::DiscreteUniform) = 0.0

function kurtosis(d::DiscreteUniform)
    n2 = span(d)^2
    -1.2 * (n2 + 1.0) / (n2 - 1.0)
end

entropy(d::DiscreteUniform) = log(span(d))

mode(d::DiscreteUniform) = d.a
modes(d::DiscreteUniform) = [d.a:d.b]


### Evaluation

cdf(d::DiscreteUniform, x::Int) = (x < d.a ? 0.0 :
                                   x > d.b ? 1.0 :
                                   (floor(Int,x) - d.a + 1.0) * d.pv)

pdf(d::DiscreteUniform, x::Int) = insupport(d, x) ? d.pv : 0.0

logpdf(d::DiscreteUniform, x::Int) = insupport(d, x) ? log(d.pv) : -Inf

quantile(d::DiscreteUniform, p::Float64) = d.a + floor(Int,p * span(d))

function mgf(d::DiscreteUniform, t::T) where {T <: Real}
    a, b = d.a, d.b
    u = b - a + 1
    t == 0 ? one(T) : (exp(t*a) * expm1(t*u)) / (u*expm1(t))
end

function cf(d::DiscreteUniform, t::T) where {T <: Real}
    a, b = d.a, d.b
    u = b - a + 1
    t == 0 ? complex(one(T)) : (im*cos(t*(a+b)/2) + sin(t*(a-b-1)/2)) / (u*sin(t/2))
end


### Sampling

rand(rng::AbstractRNG, d::DiscreteUniform) = rand(rng, d.a:d.b)

# Fit model

function fit_mle(::Type{DiscreteUniform}, x::AbstractArray{T}) where T <: Real
    if isempty(x)
        throw(ArgumentError("x cannot be empty."))
    end

    xmin = xmax = x[1]
    for i = 2:length(x)
        @inbounds xi = x[i]
        if xi < xmin
            xmin = xi
        elseif xi > xmax
            xmax = xi
        end
    end

    return DiscreteUniform(xmin, xmax)
end

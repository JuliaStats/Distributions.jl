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
struct DiscreteUniform{I} <: DiscreteUnivariateDistribution where I
    a::I
    b::I
    pv::Float64 # individual probabilities

    function DiscreteUniform{I}(a::I, b::I; check_args::Bool=true) where I <: Integer
        @check_args DiscreteUniform (a <= b)
        new{I}(a, b, one(I) / (b - a + one(I)))
    end
    function DiscreteUniform(a::Real, b::Real; kwargs...)
        a_int = a isa Integer ? a : ceil(Int,a)
        b_int = b isa Integer ? b : floor(Int, b)
        aI, bI = promote(a_int,b_int)
        I = typeof(aI)
        DiscreteUniform{I}(aI,bI; kwargs...)
    end
    DiscreteUniform(b::Real; check_args::Bool=true) = DiscreteUniform(0, b; check_args=check_args)
    DiscreteUniform() = new{Int}(0, 1, 0.5)
end

@distr_support DiscreteUniform d.a d.b

partype(::DiscreteUniform{T}) where {T<:Integer} = T

#### Conversions
convert(::Type{DiscreteUniform{T}}, a::S, b::S) where {T <: Integer, S <: Real} = DiscreteUniform(T(a), T(b))
Base.convert(::Type{DiscreteUniform{T}}, d::DiscreteUniform) where {T<:Integer} = DiscreteUniform{T}(T(d.a), T(d.b))
Base.convert(::Type{DiscreteUniform{T}}, d::DiscreteUniform{T}) where {T<:Integer} = d

### Parameters

span(d::DiscreteUniform) = d.b - d.a + 1
probval(d::DiscreteUniform) = d.pv
params(d::DiscreteUniform) = (d.a, d.b)

### Show

show(io::IO, d::DiscreteUniform) = show(io, d, (:a, :b))


### Statistics

mean(d::DiscreteUniform) = middle(d.a, d.b)

median(d::DiscreteUniform) = fld(d.a + d.b, 2)

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

pdf(d::DiscreteUniform, x::Real) = insupport(d, x) ? d.pv : zero(d.pv)
logpdf(d::DiscreteUniform, x::Real) = log(pdf(d, x))

function cdf(d::DiscreteUniform, x::Integer)
    a = d.a
    result = (x - a + 1) * d.pv
    return if x < a
        zero(result)
    elseif x >= d.b
        one(result)
    else
        result
    end
end

function quantile(d::DiscreteUniform, p::Real) 
    T = partype(d)
    iszero(p) ? d.a : d.a - one(T) + ceil(T, p * span(d))
end

function mgf(d::DiscreteUniform, t::Real)
    a, b = d.a, d.b
    u = b - a + 1
    result = (exp(t*a) * expm1(t*u)) / (u*expm1(t))
    return iszero(t) ? one(result) : result
end

function cf(d::DiscreteUniform, t::Real)
    a, b = d.a, d.b
    u = b - a + 1
    result = (im*cos(t*(a+b)/2) + sin(t*(a-b-1)/2)) / (u*sin(t/2))
    return iszero(t) ? one(result) : result
end


### Sampling

rand(rng::AbstractRNG, d::DiscreteUniform) = rand(rng, d.a:d.b)

# Fit model

function fit_mle(::Type{DiscreteUniform}, x::AbstractArray{<:Real})
    if isempty(x)
        throw(ArgumentError("data set must be non-empty."))
    end
    return DiscreteUniform(extrema(x)...)
end

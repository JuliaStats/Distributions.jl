"""
    Uniform(a,b)

The *continuous uniform distribution* over an interval ``[a, b]`` has probability density function

```math
f(x; a, b) = \\frac{1}{b - a}, \\quad a \\le x \\le b
```

```julia
Uniform()        # Uniform distribution over [0, 1]
Uniform(a, b)    # Uniform distribution over [a, b]

params(d)        # Get the parameters, i.e. (a, b)
minimum(d)       # Get the lower bound, i.e. a
maximum(d)       # Get the upper bound, i.e. b
location(d)      # Get the location parameter, i.e. a
scale(d)         # Get the scale parameter, i.e. b - a
```

External links

* [Uniform distribution (continuous) on Wikipedia](http://en.wikipedia.org/wiki/Uniform_distribution_(continuous))

"""
struct Uniform{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T

    Uniform{T}(a::T, b::T) where {T} = (@check_args(Uniform, a < b); new{T}(a, b))
end

Uniform(a::T, b::T) where {T<:Real} = Uniform{T}(a, b)
Uniform(a::Real, b::Real) = Uniform(promote(a, b)...)
Uniform(a::Integer, b::Integer) = Uniform(Float64(a), Float64(b))
Uniform() = Uniform(0.0, 1.0)

@distr_support Uniform d.a d.b

#### Conversions
convert(::Type{Uniform{T}}, a::Real, b::Real) where {T<:Real} = Uniform(T(a), T(b))
convert(::Type{Uniform{T}}, d::Uniform{S}) where {T<:Real, S<:Real} = Uniform(T(d.a), T(d.b))

#### Parameters

params(d::Uniform) = (d.a, d.b)
@inline partype(d::Uniform{T}) where {T<:Real} = T

location(d::Uniform) = d.a
scale(d::Uniform) = d.b - d.a


#### Statistics

mean(d::Uniform) = middle(d.a, d.b)
median(d::Uniform) = mean(d)
mode(d::Uniform) = mean(d)
modes(d::Uniform) = Float64[]

var(d::Uniform) = (w = d.b - d.a; w^2 / 12)

skewness(d::Uniform{T}) where {T<:Real} = zero(T)
kurtosis(d::Uniform{T}) where {T<:Real} = -6/5*one(T)

entropy(d::Uniform) = log(d.b - d.a)


#### Evaluation

pdf(d::Uniform{T}, x::Real) where {T<:Real} = insupport(d, x) ? 1 / (d.b - d.a) : zero(T)
logpdf(d::Uniform{T}, x::Real) where {T<:Real} = insupport(d, x) ? -log(d.b - d.a) : -T(Inf)

function cdf(d::Uniform{T}, x::Real) where T<:Real
    (a, b) = params(d)
    x <= a ? zero(T) :
    x >= d.b ? one(T) : (x - a) / (b - a)
end

function ccdf(d::Uniform{T}, x::Real) where T<:Real
    (a, b) = params(d)
    x <= a ? one(T) :
    x >= d.b ? zero(T) : (b - x) / (b - a)
end

quantile(d::Uniform, p::Real) = d.a + p * (d.b - d.a)
cquantile(d::Uniform, p::Real) = d.b + p * (d.a - d.b)


function mgf(d::Uniform, t::Real)
    (a, b) = params(d)
    u = (b - a) * t / 2
    u == zero(u) && return one(u)
    v = (a + b) * t / 2
    exp(v) * (sinh(u) / u)
end

function cf(d::Uniform, t::Real)
    (a, b) = params(d)
    u = (b - a) * t / 2
    u == zero(u) && return complex(one(u))
    v = (a + b) * t / 2
    cis(v) * (sin(u) / u)
end


#### Evaluation

rand(d::Uniform) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Uniform) = d.a + (d.b - d.a) * rand(rng)


#### Fitting

function fit_mle(::Type{Uniform}, x::AbstractArray{T}) where T<:Real
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

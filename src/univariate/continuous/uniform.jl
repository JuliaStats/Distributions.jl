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
    Uniform{T}(a::Real, b::Real) where {T <: Real} = new{T}(a, b)
end

function Uniform(a::T, b::T; check_args::Bool=true) where {T <: Real}
    @check_args Uniform (a < b)
    return Uniform{T}(a, b)
end

Uniform(a::Real, b::Real; check_args::Bool=true) = Uniform(promote(a, b)...; check_args=check_args)
Uniform(a::Integer, b::Integer; check_args::Bool=true) = Uniform(float(a), float(b); check_args=check_args)
Uniform() = Uniform{Float64}(0.0, 1.0)

@distr_support Uniform d.a d.b

#### Conversions
convert(::Type{Uniform{T}}, a::Real, b::Real) where {T<:Real} = Uniform(T(a), T(b))
Base.convert(::Type{Uniform{T}}, d::Uniform) where {T<:Real} = Uniform{T}(T(d.a), T(d.b))
Base.convert(::Type{Uniform{T}}, d::Uniform{T}) where {T<:Real} = d

#### Parameters

params(d::Uniform) = (d.a, d.b)
partype(::Uniform{T}) where {T<:Real} = T

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

function pdf(d::Uniform, x::Real)
    # include dependency on `x` for return type to be consistent with `cdf`
    a, b, _ = promote(d.a, d.b, x)
    val = inv(b - a)
    return insupport(d, x) ? val : zero(val)
end
function logpdf(d::Uniform, x::Real)
    # include dependency on `x` for return type to be consistent with `logcdf`
    a, b, _ = promote(d.a, d.b, x)
    val = - log(b - a)
    return insupport(d, x) ? val : oftype(val, -Inf)
end
gradlogpdf(d::Uniform, x::Real) = zero(partype(d)) / oneunit(x)

function cdf(d::Uniform, x::Real)
    a, b = params(d)
    return clamp((x - a) / (b - a), 0, 1)
end
function ccdf(d::Uniform, x::Real)
    a, b = params(d)
    return clamp((b - x) / (b - a), 0, 1)
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
function cgf_uniform_around_zero_kernel(x)
    # taylor series of (exp(x) - x - 1) / x
    T = typeof(x)
    a0 = inv(T(2))
    a1 = inv(T(6))
    a2 = inv(T(24))
    a3 = inv(T(120))
    x*@evalpoly(x, a0, a1, a2, a3)
end

function cgf(d::Uniform, t)
    # log((exp(t*b) - exp(t*a))/ (t*(b-a)))
    a,b = params(d)
    x = t*(b-a)
    if abs(x) <= sqrt(eps(float(one(x))))
        cgf_around_zero(d, t)
    else
        cgf_away_from_zero(d, t)
    end
end
function cgf_around_zero(d::Uniform, t)
    a,b = params(d)
    x = t*(b-a)
    t*a + log1p(cgf_uniform_around_zero_kernel(x))
end
function cgf_away_from_zero(d::Uniform, t)
    a,b = params(d)
    x = t*(b-a)
    logsubexp(t*b, t*a) - log(abs(x))
end

function cf(d::Uniform, t::Real)
    (a, b) = params(d)
    u = (b - a) * t / 2
    u == zero(u) && return complex(one(u))
    v = (a + b) * t / 2
    cis(v) * (sin(u) / u)
end

#### Affine transformations

Base.:+(d::Uniform, c::Real) = Uniform(d.a + c, d.b + c)
Base.:*(c::Real, d::Uniform) = Uniform(minmax(c * d.a, c * d.b)...)

#### Sampling

rand(rng::AbstractRNG, d::Uniform) = d.a + (d.b - d.a) * rand(rng)

_rand!(rng::AbstractRNG, d::Uniform, A::AbstractArray{<:Real}) =
    A .= Base.Fix1(quantile, d).(rand!(rng, A))


#### Fitting

function fit_mle(::Type{T}, x::AbstractArray{<:Real}) where {T<:Uniform}
    if isempty(x)
        throw(ArgumentError("x cannot be empty."))
    end
    return T(extrema(x)...)
end

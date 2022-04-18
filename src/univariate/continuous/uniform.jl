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
    Uniform{T}(a::T, b::T) where {T <: Real} = new{T}(a, b)
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
    val = inv(d.b - d.a)
    return insupport(d, x) ? val : zero(val)
end
function logpdf(d::Uniform, x::Real)
    diff = d.b - d.a
    return insupport(d, x) ? -log(diff) : log(zero(diff))
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

function cf(d::Uniform, t::Real)
    (a, b) = params(d)
    u = (b - a) * t / 2
    u == zero(u) && return complex(one(u))
    v = (a + b) * t / 2
    cis(v) * (sin(u) / u)
end

#### Fast path for `loglikelihood`

function loglikelihood(d::Uniform, x::AbstractArray{<:Real})
    a, b = params(d)
    diff = b - a
    return all(x -> a <= x <= b, x) ? -length(x) * log(diff) : log(zero(diff))
end

#### Affine transformations

Base.:+(d::Uniform, c::Real) = Uniform(d.a + c, d.b + c)
Base.:*(c::Real, d::Uniform) = Uniform(minmax(c * d.a, c * d.b)...)

#### Sampling

rand(rng::AbstractRNG, d::Uniform) = d.a + (d.b - d.a) * rand(rng)

_rand!(rng::AbstractRNG, d::Uniform, A::AbstractArray{<:Real}) =
    A .= quantile.(d, rand!(rng, A))


#### Fitting

function fit_mle(::Type{<:Uniform}, x::AbstractArray{T}) where T<:Real
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

# ChainRules definitions
# Added to ensure that derivatives for values at boundaries and not in support are zero
# and to imporve performance
# Ref: https://github.com/JuliaStats/Distributions.jl/pull/1459

## logpdf
function ChainRulesCore.frule((_, Δd, _), ::typeof(logpdf), d::Uniform, x::Real)
    # Compute log probability
    a, b = params(d)
    diff = b - a
    Ω = a <= x <= b ? -log(diff) : log(zero(diff))

    # Compute tangent
    # Return zero for values at the boundary or not in the support
    # Ref: https://github.com/JuliaStats/Distributions.jl/pull/1459
    Δdiff = Δd.a - Δd.b
    ΔΩ = (a < x < b ? Δdiff : zero(Δdiff)) / diff

    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(logpdf), d::Uniform, x::Real)
    # Compute log probability
    a, b = params(d)
    diff = b - a
    Ω = a <= x <= b ? -log(diff) : log(zero(diff))

    # Define pullback
    # Return zero for values at the boundary or not in the support
    # Ref: https://github.com/JuliaStats/Distributions.jl/pull/1459
    insidesupport = a < x < b
    function logpdf_Uniform_pullback(Δ)
        Δa = Δ / diff
        Δd = if insidesupport
            ChainRulesCore.Tangent{typeof(d)}(; a=Δa, b=-Δa)
        else
            ChainRulesCore.Tangent{typeof(d)}(; a=zero(Δa), b=zero(Δa))
        end
        return ChainRulesCore.NoTangent(), Δd, ChainRulesCore.ZeroTangent()
    end

    return Ω, logpdf_Uniform_pullback
end

## loglikelihood
function ChainRulesCore.frule((_, Δd, _), ::typeof(loglikelihood), d::Uniform, x::AbstractArray{<:Real})
    # Compute log likelihood
    a, b = params(d)
    n = length(x)
    count_insidesupport = count(x -> a < x < b, x) # used below
    all_insupport = count_insidesupport == n || all(x -> a <= x <= b, x)
    diff = b - a
    Ω = all_insupport ? -n * log(diff) : log(zero(diff))

    # Compute tangent
    # Return zero for values at the boundary or not in the support
    # Ref: https://github.com/JuliaStats/Distributions.jl/pull/1459
    ΔΩ = count_insidesupport * (Δd.a - Δd.b) / diff

    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(loglikelihood), d::Uniform, x::AbstractArray{<:Real})
    # Compute log likelihood
    a, b = params(d)
    n = length(x)
    count_insidesupport = count(x -> a < x < b, x) # used below
    all_insupport = count_insidesupport == n || all(x -> a <= x <= b, x)
    diff = b - a
    Ω = all_insupport ? -n * log(diff) : log(zero(diff))

    # Define pullback
    # Return zero for values at the boundary or not in the support
    # Ref: https://github.com/JuliaStats/Distributions.jl/pull/1459
    function loglikelihood_Uniform_pullback(Δ)
        Δa = count_insidesupport * Δ / diff
        Δd = ChainRulesCore.Tangent{typeof(d)}(; a=Δa, b=-Δa)
        return ChainRulesCore.NoTangent(), Δd, ChainRulesCore.ZeroTangent()
    end

    return Ω, loglikelihood_Uniform_pullback
end

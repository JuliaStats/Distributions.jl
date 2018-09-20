"""
    Laplace(μ,θ)

The *Laplace distribution* with location `μ` and scale `θ` has probability density function

```math
f(x; \\mu, \\beta) = \\frac{1}{2 \\beta} \\exp \\left(- \\frac{|x - \\mu|}{\\beta} \\right)
```

```julia
Laplace()       # Laplace distribution with zero location and unit scale, i.e. Laplace(0, 1)
Laplace(u)      # Laplace distribution with location u and unit scale, i.e. Laplace(u, 1)
Laplace(u, b)   # Laplace distribution with location u ans scale b

params(d)       # Get the parameters, i.e. (u, b)
location(d)     # Get the location parameter, i.e. u
scale(d)        # Get the scale parameter, i.e. b
```

External links

* [Laplace distribution on Wikipedia](http://en.wikipedia.org/wiki/Laplace_distribution)

"""
struct Laplace{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    θ::T

    Laplace{T}(μ::T, θ::T) where {T} = (@check_args(Laplace, θ > zero(θ)); new{T}(μ, θ))
end

Laplace(μ::T, θ::T) where {T<:Real} = Laplace{T}(μ, θ)
Laplace(μ::Real, θ::Real) = Laplace(promote(μ, θ)...)
Laplace(μ::Integer, θ::Integer) = Laplace(Float64(μ), Float64(θ))
Laplace(μ::Real) = Laplace(μ, 1.0)
Laplace() = Laplace(0.0, 1.0)

const Biexponential = Laplace

@distr_support Laplace -Inf Inf

#### Conversions
function convert(::Type{Laplace{T}}, μ::S, θ::S) where {T <: Real, S <: Real}
    Laplace(T(μ), T(θ))
end
function convert(::Type{Laplace{T}}, d::Laplace{S}) where {T <: Real, S <: Real}
    Laplace(T(d.μ), T(d.θ))
end


#### Parameters

location(d::Laplace) = d.μ
scale(d::Laplace) = d.θ
params(d::Laplace) = (d.μ, d.θ)
@inline partype(d::Laplace{T}) where {T<:Real} = T


#### Statistics

mean(d::Laplace) = d.μ
median(d::Laplace) = d.μ
mode(d::Laplace) = d.μ

var(d::Laplace) = 2d.θ^2
std(d::Laplace) = sqrt2 * d.θ
skewness(d::Laplace{T}) where {T<:Real} = zero(T)
kurtosis(d::Laplace{T}) where {T<:Real} = 3one(T)

entropy(d::Laplace) = log(2d.θ) + 1


#### Evaluations

zval(d::Laplace, x::Real) = (x - d.μ) / d.θ
xval(d::Laplace, z::Real) = d.μ + z * d.θ

pdf(d::Laplace, x::Real) = exp(-abs(zval(d, x))) / 2scale(d)
logpdf(d::Laplace, x::Real) = - (abs(zval(d, x)) + log(2scale(d)))

cdf(d::Laplace, x::Real) = (z = zval(d, x); z < 0 ? exp(z)/2 : 1 - exp(-z)/2)
ccdf(d::Laplace, x::Real) = (z = zval(d, x); z > 0 ? exp(-z)/2 : 1 - exp(z)/2)
logcdf(d::Laplace, x::Real) = (z = zval(d, x); z < 0 ? loghalf + z : loghalf + log2mexp(-z))
logccdf(d::Laplace, x::Real) = (z = zval(d, x); z > 0 ? loghalf - z : loghalf + log2mexp(z))

quantile(d::Laplace, p::Real) = p < 1/2 ? xval(d, log(2p)) : xval(d, -log(2(1 - p)))
cquantile(d::Laplace, p::Real) = p > 1/2 ? xval(d, log(2(1 - p))) : xval(d, -log(2p))
invlogcdf(d::Laplace, lp::Real) = lp < loghalf ? xval(d, logtwo + lp) : xval(d, -(logtwo + log1mexp(lp)))
invlogccdf(d::Laplace, lp::Real) = lp > loghalf ? xval(d, logtwo + log1mexp(lp)) : xval(d, -(logtwo + lp))

function gradlogpdf(d::Laplace, x::Real)
    μ, θ = params(d)
    x == μ && error("Gradient is undefined at the location point")
    g = 1 / θ
    x > μ ? -g : g
end

function mgf(d::Laplace, t::Real)
    st = d.θ * t
    exp(t * d.μ) / ((1 - st) * (1 + st))
end
function cf(d::Laplace, t::Real)
    st = d.θ * t
    cis(t * d.μ) / (1+st*st)
end


#### Sampling

rand(d::Laplace) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Laplace) = d.μ + d.θ*randexp(rng)*ifelse(rand(rng, Bool), 1, -1)


#### Fitting

function fit_mle(::Type{Laplace}, x::Array)
    xc = copy(x)
    a = median!(xc)
    Laplace(a, StatsBase.mad!(xc, center=a))
end

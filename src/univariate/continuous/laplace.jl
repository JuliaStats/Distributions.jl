"""
    Laplace(μ,θ)

The *Laplace distribution* with location `μ` and scale `θ` has probability density function

```math
f(x; \\mu, \\theta) = \\frac{1}{2 \\theta} \\exp \\left(- \\frac{|x - \\mu|}{\\theta} \\right)
```

```julia
Laplace()       # Laplace distribution with zero location and unit scale, i.e. Laplace(0, 1)
Laplace(μ)      # Laplace distribution with location μ and unit scale, i.e. Laplace(μ, 1)
Laplace(μ, θ)   # Laplace distribution with location μ and scale θ

params(d)       # Get the parameters, i.e., (μ, θ)
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. θ
```

External links

* [Laplace distribution on Wikipedia](http://en.wikipedia.org/wiki/Laplace_distribution)

"""
struct Laplace{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    θ::T
    Laplace{T}(µ::T, θ::T) where {T} = new{T}(µ, θ)
end

function Laplace(μ::T, θ::T; check_args::Bool=true) where {T <: Real}
    @check_args Laplace (θ, θ > zero(θ))
    return Laplace{T}(μ, θ)
end

Laplace(μ::Real, θ::Real; check_args::Bool=true) = Laplace(promote(μ, θ)...; check_args=check_args)
Laplace(μ::Integer, θ::Integer; check_args::Bool=true) = Laplace(float(μ), float(θ); check_args=check_args)
Laplace(μ::Real=0.0) = Laplace(μ, one(μ); check_args=false)

const Biexponential = Laplace

@distr_support Laplace -Inf Inf

#### Conversions
function convert(::Type{Laplace{T}}, μ::S, θ::S) where {T <: Real, S <: Real}
    Laplace(T(μ), T(θ))
end
function Base.convert(::Type{Laplace{T}}, d::Laplace) where {T<:Real}
    Laplace{T}(T(d.μ), T(d.θ))
end
Base.convert(::Type{Laplace{T}}, d::Laplace{T}) where {T<:Real} = d

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
        
function kldivergence(p::Laplace, q::Laplace)
    pμ, pθ = params(p)
    qμ, qθ = params(q)
    r = abs(pμ - qμ)
    return (pθ * exp(-r / pθ) + r) / qθ + log(qθ / pθ) - 1
end

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
function cgf(d::Laplace, t)
    μ, θ = params(d)
    t*μ - log1p(-(θ*t)^2)
end
function cf(d::Laplace, t::Real)
    st = d.θ * t
    cis(t * d.μ) / (1+st*st)
end

#### Affine transformations

Base.:+(d::Laplace, c::Real) = Laplace(d.μ + c, d.θ)
Base.:*(c::Real, d::Laplace) = Laplace(c * d.μ, abs(c) * d.θ)

#### Sampling

rand(rng::AbstractRNG, d::Laplace) =
    d.μ + d.θ*randexp(rng)*ifelse(rand(rng, Bool), 1, -1)


#### Fitting

function fit_mle(::Type{<:Laplace}, x::AbstractArray{<:Real})
    xc = similar(x)
    copyto!(xc, x)
    m = median!(xc)
    xc .= abs.(x .- m)
    return Laplace(m, mean(xc))
end

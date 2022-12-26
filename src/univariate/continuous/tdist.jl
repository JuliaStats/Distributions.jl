"""
    TDist(ν)

The *Students T distribution* with `ν` degrees of freedom has probability density function

```math
f(x; \\nu) = \\frac{1}{\\sqrt{\\nu} B(1/2, \\nu/2)}
\\left( 1 + \\frac{x^2}{\\nu} \\right)^{-\\frac{\\nu + 1}{2}}
```

```julia
TDist(d)      # t-distribution with ν degrees of freedom

params(d)     # Get the parameters, i.e. (ν,)
dof(d)        # Get the degrees of freedom, i.e. ν
```

External links

[Student's T distribution on Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-distribution)

"""
struct TDist{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    TDist{T}(ν::T) where {T <: Real} = new{T}(ν)
end

function TDist(ν::Real; check_args::Bool=true)
    @check_args TDist (ν, ν > zero(ν))
    return TDist{typeof(ν)}(ν)
end

TDist(ν::Integer; check_args::Bool=true) = TDist(float(ν); check_args=check_args)

@distr_support TDist -Inf Inf

#### Conversions
convert(::Type{TDist{T}}, ν::Real) where {T<:Real} = TDist(T(ν))
Base.convert(::Type{TDist{T}}, d::TDist) where {T<:Real} = TDist{T}(T(d.ν))
Base.convert(::Type{TDist{T}}, d::TDist{T}) where {T<:Real} = d

#### Parameters

dof(d::TDist) = d.ν
params(d::TDist) = (d.ν,)
@inline partype(d::TDist{T}) where {T<:Real} = T


#### Statistics

mean(d::TDist{T}) where {T<:Real} = d.ν > 1 ? zero(T) : T(NaN)
median(d::TDist{T}) where {T<:Real} = zero(T)
mode(d::TDist{T}) where {T<:Real} = zero(T)

function var(d::TDist{T}) where T<:Real
    ν = d.ν
    isinf(ν) && return one(T)
    ν > 2 ? ν / (ν - 2) :
    ν > 1 ? T(Inf) : T(NaN)
end

skewness(d::TDist{T}) where {T<:Real} = d.ν > 3 ? zero(T) : T(NaN)

function kurtosis(d::TDist{T}) where T<:Real
    ν = d.ν
    ν > 4 ? 6 / (ν - 4) :
    ν > 2 ? T(Inf) : T(NaN)
end

function entropy(d::TDist{T}) where T <: Real
    isinf(d.ν) && return entropy( Normal(zero(T), one(T)) )
    h = d.ν/2
    h1 = h + 1//2
    h1 * (digamma(h1) - digamma(h)) + log(d.ν)/2 + logbeta(h, 1//2)
end


#### Evaluation & Sampling

@_delegate_statsfuns TDist tdist ν

function rand(rng::AbstractRNG, d::TDist)
    ν = d.ν
    z = sqrt(rand(rng, Chisq{typeof(ν)}(ν)) / ν)
    return randn(rng) / (isinf(ν) ? one(z) : z)
end

function cf(d::TDist{T}, t::Real) where T <: Real
    isinf(d.ν) && return cf(Normal(zero(T), one(T)), t)
    t == 0 && return complex(1)
    h = d.ν/2
    q = d.ν/4
    complex(2(q*t^2)^q * besselk(h, sqrt(d.ν) * abs(t)) / gamma(h))
end

gradlogpdf(d::TDist{T}, x::Real) where {T<:Real} = isinf(d.ν) ? gradlogpdf(Normal(zero(T), one(T)), x) : -((d.ν + 1) * x) / (x^2 + d.ν)

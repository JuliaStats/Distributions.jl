doc"""
    TDist(ν)

The *Students T distribution* with `ν` degrees of freedom has probability density function

$f(x; d) = \frac{1}{\sqrt{d} B(1/2, d/2)}
\left( 1 + \frac{x^2}{d} \right)^{-\frac{d + 1}{2}}$

```julia
TDist(d)      # t-distribution with d degrees of freedom

params(d)     # Get the parameters, i.e. (d,)
dof(d)        # Get the degrees of freedom, i.e. d
```

External links

[Student's T distribution on Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-distribution)

"""
immutable TDist{T<:Real} <: ContinuousUnivariateDistribution
    ν::T

    TDist(ν::T) = (@check_args(TDist, ν > zero(ν)); new(ν))
end

TDist{T<:Real}(ν::T) = TDist{T}(ν)
TDist(ν::Integer) = TDist(Float64(ν))

@distr_support TDist -Inf Inf

#### Conversions
convert{T<:Real}(::Type{TDist{T}}, ν::Real) = TDist(T(ν))
convert{T<:Real, S<:Real}(::Type{TDist{T}}, d::TDist{S}) = TDist(T(d.ν))

#### Parameters

dof(d::TDist) = d.ν
params(d::TDist) = (d.ν,)
@inline partype{T<:Real}(d::TDist{T}) = T


#### Statistics

mean{T<:Real}(d::TDist{T}) = d.ν > 1 ? zero(T) : T(NaN)
median{T<:Real}(d::TDist{T}) = zero(T)
mode{T<:Real}(d::TDist{T}) = zero(T)

function var{T<:Real}(d::TDist{T})
    ν = d.ν
    ν > 2 ? ν / (ν - 2) :
    ν > 1 ? T(Inf) : T(NaN)
end

skewness{T<:Real}(d::TDist{T}) = d.ν > 3 ? zero(T) : T(NaN)

function kurtosis{T<:Real}(d::TDist{T})
    ν = d.ν
    ν > 4 ? 6 / (ν - 4) :
    ν > 2 ? T(Inf) : T(NaN)
end

function entropy(d::TDist)
    h = d.ν/2
    h1 = h + 1//2
    h1 * (digamma(h1) - digamma(h)) + log(d.ν)/2 + lbeta(h, 0.5)
end


#### Evaluation & Sampling

@_delegate_statsfuns TDist tdist ν

rand(d::TDist) = StatsFuns.RFunctions.tdistrand(d.ν)

function cf(d::TDist, t::Real)
    t == 0 && return complex(1)
    h = d.ν/2
    q = d.ν/4
    complex(2(q*t^2)^q * besselk(h, sqrt(d.ν) * abs(t)) / gamma(h))
end

gradlogpdf(d::TDist, x::Real) = -((d.ν + 1) * x) / (x^2 + d.ν)

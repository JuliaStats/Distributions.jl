# Modified from code provided by "Doomphoenix Qxz" on discourse.julialang.com
"""
    StudentizedRange(ν, k)

The *studentized range distribution* has probability density function:

```math
f(q; k, \\nu) = \\frac{\\sqrt{2\\pi}k(k - 1)\\nu^{\\nu/2}}{\\Gamma{\\left(\\frac{\\nu}{2}\\right)}2^{\\nu/2 - 1}} \\int_{0}^{\\infty} {x^{\\nu}\\phi(\\sqrt{\\nu}x)} \\left[\\int_{-\\infty}^{\\infty} {\\phi(u)\\phi(u - qx)[\\Phi(u) - \\Phi(u - qx)]^{k - 2}}du\\right]dx
```

where

```math
\\begin{aligned}
\\Phi(x) &= \\frac{1 + erf(\\frac{x}{\\sqrt{2}})}{2} &&(\\text{Normal Distribution CDF})\\\\
\\phi(x) &= \\Phi'(x) &&(\\text{Normal Distribution PDF})
\\end{aligned}
```

```julia
StudentizedRange(ν, k)     # Studentized Range Distribution with parameters ν and k

params(d)        # Get the parameters, i.e. (ν, k)
```

External links

* [Studentized range distribution on Wikipedia](http://en.wikipedia.org/wiki/Studentized_range_distribution)
"""
struct StudentizedRange{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    k::T
    StudentizedRange{T}(ν::T, k::T) where {T <: Real} = new{T}(ν, k)
end

function StudentizedRange(ν::T, k::T; check_args::Bool=true) where {T <: Real}
    @check_args StudentizedRange (ν, ν > zero(ν)) (k, k > one(k))
    return StudentizedRange{T}(ν, k)
end

StudentizedRange(ν::Integer, k::Integer; check_args::Bool=true) = StudentizedRange(float(ν), float(k); check_args=check_args)
StudentizedRange(ν::Real, k::Real; check_args::Bool=true) = StudentizedRange(promote(ν, k)...; check_args=check_args)

@distr_support StudentizedRange 0.0 Inf


###  Conversions

function convert(::Type{StudentizedRange{T}}, ν::S, k::S) where {T <: Real, S <: Real}
    StudentizedRange(T(ν), T(k))
end

function Base.convert(::Type{StudentizedRange{T}}, d::StudentizedRange) where {T<:Real}
    StudentizedRange{T}(T(d.ν), T(d.k))
end
Base.convert(::Type{StudentizedRange{T}}, d::StudentizedRange{T}) where {T<:Real} = d

### Parameters
params(d::StudentizedRange) = (d.ν, d.k)
@inline partype(d::StudentizedRange{T}) where {T <: Real} = T


### Evaluation & Sampling

@_delegate_statsfuns StudentizedRange srdist k ν

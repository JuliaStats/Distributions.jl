# Modified from code provided by "Doomphoenix Qxz" on discourse.julialang.com
"""
    StudentizedRange(ν, k)

The *studentized range distribution* has probability density function:

```math
f(q; k, \\nu) = \\frac{\\sqrt{2\\pi}k(k - 1)\\nu^{\\nu/2}}{\\Gamma{(\\frac{\\nu}{2})}2^{\\nu/2 - 1}} \\int_{0}^{\\infty} {x^{\\nu}\\phi(\\sqrt{\\nu}x)} [\\int_{-\\infty}^{\\infty} {\\phi(u)\\phi(u - qx)(\\Phi(u) - \\Phi(u - qx))^{k - 2}du]dx

where

\\Phi(x) = \\frac{1 + erf(\\frac{x}{\\sqrt{2}})}{2} (Normal Distribution CDF)
\\phi(x) = \\Phi'(x) (Normal Distribution PDF)
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

    function StudentizedRange{T}(ν::T, k::T) where T
        @check_args(StudentizedRange, ν > zero(ν) && k > one(k))
        new{T}(ν, k)
    end
end

StudentizedRange(ν::T, k::T) where {T<:Real} = StudentizedRange{T}(ν, k)
StudentizedRange(ν::Integer, k::Integer) = StudentizedRange(Float64(ν), Float64(k))
StudentizedRange(ν::Real, k::Real) = StudentizedRange(promote(ν, k)...)

@distr_support StudentizedRange 0.0 Inf


###  Conversions

function convert(::Type{StudentizedRange{T}}, ν::S, k::S) where {T <: Real, S <: Real}
    StudentizedRange(T(ν), T(k))
end

function convert(::Type{StudentizedRange{T}}, d::StudentizedRange{S}) where {T <: Real, S <: Real}
    StudentizedRange(T(d.ν), T(d.k))
end


### Parameters
params(d::StudentizedRange) = (d.ν, d.k)
@inline partype(d::StudentizedRange{T}) where {T <: Real} = T


### Evaluation & Sampling

@_delegate_statsfuns StudentizedRange srdist k ν

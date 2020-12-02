"""

    ZipfMandelbrot(N, q, s)

The *Zipf-Mandelbrot law* with shape parameters `q` and `s` and number of categories `N` has probability mass function
```math
P(X = k) = \\frac{1/(k+q)^s}{H_{N,q,s}}, \\quad \\text{ for } k = 1,\\ldots,N,
```
where H_{N,q,s} is given by
```math
H_{N,q,s} = \\sum_{i=1}^N \\frac{1}{(i+q)^s}.
```

```julia
ZipfMandelbrot(N, q, s)    # Zipf-Mandelbrot law with parameters N, q, s 

params(d)    # Get the parameters, i.e. (N, q, s)
```

External links:
 * [Zipf-Mandelbrot law on Wikipedia](https://en.wikipedia.org/wiki/Zipf%E2%80%93Mandelbrot_law)
"""

struct ZipfMandelbrot{T<:Real} <: DiscreteUnivariateDistribution
    N::Int
    q::T
    s::T
    ZipfMandelbrot{T}(N::Int, q::T, s::T) where {T} = new{T}(N, q, s)
end

function ZipfMandelbrot(N::Int, q::T, s::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZipfMandelbrot, N >= one(N) && q >= zero(q) && s > zero(s))
    return ZipfMandelbrot{T}(N, q, s)
end

ZipfMandelbrot(N::Int, q::Real, s::Real) = ZipfMandelbrot(N, promote(q, s)...)
ZipfMandelbrot(N::Int, q::Int, s::Int) = ZipfMandelbrot(N, float(q), float(s))
ZipfMandelbrot(N::Real, q, s) = ZipfMandelbrot(Int(N), q, s)

@distr_support ZipfMandelbrot 1 d.N

#### Conversions

function convert(::Type{ZipfMandelbrot{T}}, N::S, q::S, s::S) where {T <: Real, S <: Real}
    ZipfMandelbrot(N, T(q), T(s))
end
function convert(::Type{ZipfMandelbrot{T}}, d::ZipfMandelbrot{S}) where {T <: Real, S <: Real}
    ZipfMandelbrot(d.N, T(d.q), T(d.s), check_args=false)
end

#### Parameters

params(d::ZipfMandelbrot) = (d.N, d.q, d.s)
partype(d::ZipfMandelbrot{T}) where {T<:Real} = T

#### Statistics

H(N, q, s) = sum(1 / (i+q)^s for i = 1:N)

mean(d::ZipfMandelbrot) = H(d.N, d.q, d.s-1) / H(d.N, d.q, d.s) - d.q
mode(d::ZipfMandelbrot) = 1

function entropy(d::ZipfMandelbrot)
    d.s * sum(log(i+d.q) / (i+d.q)^d.s for i = 1:d.N) / H(d.N, d.q, d.s) + log(H(d.N, d.q, d.s))
end

#### Evaluation

pdf(d::ZipfMandelbrot, k::Int) = 1 / ((k+d.q)^d.s * H(d.N, d.q, d.s))
cdf(d::ZipfMandelbrot, k::Int) = H(k, d.q, d.s) / H(d.N, d.q, d.s)

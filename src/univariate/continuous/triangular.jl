"""
    TriangularDist(a,b,c)

The *triangular distribution* with lower limit `a`, upper limit `b` and mode `c` has probability density function

```math
f(x; a, b, c)= \\begin{cases}
        0 & \\mathrm{for\\ } x < a, \\\\
        \\frac{2(x-a)}{(b-a)(c-a)} & \\mathrm{for\\ } a \\le x \\leq c, \\\\[4pt]
        \\frac{2(b-x)}{(b-a)(b-c)} & \\mathrm{for\\ } c < x \\le b, \\\\[4pt]
        0 & \\mathrm{for\\ } b < x,
        \\end{cases}
```

```julia
TriangularDist(a, b)        # Triangular distribution with lower limit a, upper limit b, and mode (a+b)/2
TriangularDist(a, b, c)     # Triangular distribution with lower limit a, upper limit b, and mode c

params(d)       # Get the parameters, i.e. (a, b, c)
minimum(d)      # Get the lower bound, i.e. a
maximum(d)      # Get the upper bound, i.e. b
mode(d)         # Get the mode, i.e. c
```

External links

* [Triangular distribution on Wikipedia](http://en.wikipedia.org/wiki/Triangular_distribution)

"""
struct TriangularDist{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    c::T
    TriangularDist{T}(a::T, b::T, c::T) where {T <: Real} = new{T}(a, b, c)
end

function TriangularDist(a::T, b::T, c::T; check_args=true) where {T <: Real}
    check_args && @check_args(TriangularDist, a <= c <= b)
    return TriangularDist{T}(a, b, c)
end

TriangularDist(a::T, b::T) where {T <: Real} = TriangularDist(a, b, middle(a, b))

TriangularDist(a::Real, b::Real, c::Real) = TriangularDist(promote(a, b, c)...)
TriangularDist(a::Integer, b::Integer, c::Integer) = TriangularDist(float(a), float(b), float(c))
TriangularDist(a::Real, b::Real) = TriangularDist(promote(a, b)...)
TriangularDist(a::Integer, b::Integer) = TriangularDist(float(a), float(b))

@distr_support TriangularDist d.a d.b

#### Conversions
convert(::Type{TriangularDist{T}}, a::Real, b::Real, c::Real) where {T<:Real} = TriangularDist(T(a), T(b), T(c))
convert(::Type{TriangularDist{T}}, d::TriangularDist{S}) where {T<:Real, S<:Real} = TriangularDist(T(d.a), T(d.b), T(d.c), check_args=false)

#### Parameters

params(d::TriangularDist) = (d.a, d.b, d.c)
partype(::TriangularDist{T}) where {T<:Real} = T


#### Statistics

mode(d::TriangularDist) = d.c

mean(d::TriangularDist) = (d.a + d.b + d.c) / 3

function median(d::TriangularDist)
    (a, b, c) = params(d)
    m = middle(a, b)
    c >= m ? a + sqrt((b - a) * (c - a)/2) :
             b - sqrt((b - a) * (b - c)/2)
end

_pretvar(a::Real, b::Real, c::Real) = a*a + b*b + c*c - a*b - a*c - b*c

function var(d::TriangularDist)
    (a, b, c) = params(d)
    _pretvar(a, b, c) / 18
end

function skewness(d::TriangularDist{T}) where T<:Real
    (a, b, c) = params(d)
    sqrt2 * (a + b - 2c) * (2a - b - c) * (a - 2b + c) / ( 5 * _pretvar(a, b, c)^(T(3)/2) )
end

kurtosis(d::TriangularDist{T}) where {T<:Real} = T(-3)/5

entropy(d::TriangularDist{T}) where {T<:Real} = one(T)/2 + log((d.b - d.a) / 2)


#### Evaluation

function pdf(d::TriangularDist{T}, x::Real) where T<:Real
    (a, b, c) = params(d)
    x <= a ? zero(T) :
    x <  c ? 2 * (x - a) / ((b - a) * (c - a)) :
    x == c ? 2 / (b - a) :
    x <= b ? 2 * (b - x) / ((b - a) * (b - c)) : zero(T)
end

function cdf(d::TriangularDist{T}, x::Real) where T<:Real
    (a, b, c) = params(d)
    x <= a ? zero(T) :
    x <  c ? (x - a)^2 / ((b - a) * (c - a)) :
    x == c ? (c - a) / (b - a) :
    x <= b ? 1 - (b - x)^2 / ((b - a) * (b - c)) : one(T)
end

function quantile(d::TriangularDist, p::Real)
    (a, b, c) = params(d)
    c_m_a = c - a
    b_m_a = b - a
    rl = c_m_a / b_m_a
    p <= rl ? a + sqrt(b_m_a * c_m_a * p) :
              b - sqrt(b_m_a * (b - c) * (1 - p))
end

function mgf(d::TriangularDist{T}, t::Real) where T<:Real
    if t == zero(t)
        return one(T)
    else
        (a, b, c) = params(d)
        u = (b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)
        v = (b - a) * (c - a) * (b - c) * t^2
        return 2u / v
    end
end

function cf(d::TriangularDist{T}, t::Real) where T<:Real
    # Is this correct?
    if t == zero(t)
        return one(Complex{T})
    else
        (a, b, c) = params(d)
        u = (b - c) * cis(a * t) - (b - a) * cis(c * t) + (c - a) * cis(b * t)
        v = (b - a) * (c - a) * (b - c) * t^2
        return -2u / v
    end
end


#### Sampling

function rand(rng::AbstractRNG, d::TriangularDist)
    (a, b, c) = params(d)
    b_m_a = b - a
    u = rand(rng)
    b_m_a * u < (c - a) ? d.a + sqrt(u * b_m_a * (c - a)) :
                          d.b - sqrt((1 - u) * b_m_a * (b - c))
end

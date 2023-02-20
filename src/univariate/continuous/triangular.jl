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

function TriangularDist(a::T, b::T, c::T; check_args::Bool=true) where {T <: Real}
    @check_args TriangularDist (a <= c <= b)
    return TriangularDist{T}(a, b, c)
end

TriangularDist(a::Real, b::Real, c::Real; check_args::Bool=true) = TriangularDist(promote(a, b, c)...; check_args=check_args)
function TriangularDist(a::Integer, b::Integer, c::Integer; check_args::Bool=true)
    TriangularDist(float(a), float(b), float(c); check_args=check_args)
end

TriangularDist(a::Real, b::Real) = TriangularDist(a, b, middle(a, b); check_args=false)

@distr_support TriangularDist d.a d.b

#### Conversions
convert(::Type{TriangularDist{T}}, a::Real, b::Real, c::Real) where {T<:Real} = TriangularDist(T(a), T(b), T(c))
Base.convert(::Type{TriangularDist{T}}, d::TriangularDist) where {T<:Real} = TriangularDist{T}(T(d.a), T(d.b), T(d.c))
Base.convert(::Type{TriangularDist{T}}, d::TriangularDist{T}) where {T<:Real} = d

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

function pdf(d::TriangularDist, x::Real)
    a, b, c = params(d)
    res = if x < c
        2 * (x - a) / ((b - a) * (c - a))
    elseif x > c
        2 * (b - x) / ((b - a) * (b - c))
    else
        # Handle x == c separately to avoid `NaN` if `c == a` or `c == b`
        oftype(x - a, 2) / (b - a)
    end
    return insupport(d, x) ? res : zero(res)
end
logpdf(d::TriangularDist, x::Real) = log(pdf(d, x))

function cdf(d::TriangularDist, x::Real)
    a, b, c = params(d)
    if x < c
        res = (x - a)^2 / ((b - a) * (c - a))
        return x < a ? zero(res) : res
    else
        res = 1 - (b - x)^2 / ((b - a) * (b - c))
        return x ≥ b ? one(res) : res
    end
end

function quantile(d::TriangularDist, p::Real)
    (a, b, c) = params(d)
    c_m_a = c - a
    b_m_a = b - a
    rl = c_m_a / b_m_a
    p <= rl ? a + sqrt(b_m_a * c_m_a * p) :
              b - sqrt(b_m_a * (b - c) * (1 - p))
end

_expm1(x::Number) = expm1(x)
if VERSION < v"1.7.0-DEV.1172"
    # expm1(::Float16) is not defined in older Julia versions
    _expm1(x::Float16) = Float16(expm1(Float32(x)))
    function _expm1(x::Complex{Float16})
        xr, xi = reim(x)
        yr, yi = reim(expm1(complex(Float32(xr), Float32(xi))))
        return complex(Float16(yr), Float16(yi))
    end
end

"""
    _phi2(x::Real)

Compute
```math
2 (exp(x) - 1 - x) / x^2
```
with the correct limit at ``x = 0``.
"""
function _phi2(x::Real)
    res = 2 * (_expm1(x) - x) / x^2
    return iszero(x) ? one(res) : res
end
function mgf(d::TriangularDist, t::Real)
    a, b, c = params(d)
    # In principle, only two branches (degenerate + non-degenerate case) are needed
    # But writing out all four cases will avoid unnecessary computations
    if a < c
        if c < b
            # Case: a < c < b
            return exp(c * t) * ((c - a) * _phi2((a - c) * t) + (b - c) * _phi2((b - c) * t)) / (b - a)
        else
            # Case: a < c = b
            return exp(c * t) * _phi2((a - c) * t)
        end
    elseif c < b
        # Case: a = c < b
        return exp(c * t) * _phi2((b - c) * t)
    else
        # Case: a = c = b
        return exp(c * t)
    end
end

"""
    _cisphi2(x::Real)

Compute
```math
- 2 (exp(x im) - 1 - x im) / x^2
```
with the correct limit at ``x = 0``.
"""
function _cisphi2(x::Real)
    z = x * im
    res = -2 * (_expm1(z) - z) / x^2
    return iszero(x) ? one(res) : res
end
function cf(d::TriangularDist, t::Real)
    a, b, c = params(d)
    # In principle, only two branches (degenerate + non-degenerate case) are needed
    # But writing out all four cases will avoid unnecessary computations
    if a < c
        if c < b
            # Case: a < c < b
            return cis(c * t) * ((c - a) * _cisphi2((a - c) * t) + (b - c) * _cisphi2((b - c) * t)) / (b - a)
        else
            # Case: a < c = b
            return cis(c * t) * _cisphi2((a - c) * t)
        end
    elseif c < b
        # Case: a = c < b
        return cis(c * t) * _cisphi2((b - c) * t)
    else
        # Case: a = c = b
        return cis(c * t)
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

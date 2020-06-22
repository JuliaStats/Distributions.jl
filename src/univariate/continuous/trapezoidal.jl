"""
    TrapezoidalDist(a,b,c)

The *trapezoidal distribution* with lower limit `a`, upper limit `d`, and a plateau between `b` and `c`. The probability density function is

```math
f(x; a, b, c)= \\begin{cases}
                   \\frac{2}{d+c-a-b}\\frac{x-a}{b-a}  & \\text{for } a\\le x < b \\\\
                   \\frac{2}{d+c-a-b}  & \\text{for } b\\le x < c \\\\
                   \\frac{2}{d+c-a-b}\\frac{d-x}{d-c}  & \\text{for } c\\le x \\le d
               \\end{cases}
```

```julia
TrapezoidalDist(a, b, c, d)

params(d)       # Get the parameters, i.e. (a, b, c, d)
minimum(d)      # Get the lower bound, i.e. a
maximum(d)      # Get the upper bound, i.e. d
```

External links

* [Trapezoidal distribution on Wikipedia](https://en.wikipedia.org/wiki/Trapezoidal_distribution)

"""
struct TrapezoidalDist{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    c::T
    d::T
    TrapezoidalDist{T}(a::T, b::T, c::T, d::T) where {T <: Real} = new{T}(a, b, c, d)
end


function TrapezoidalDist(a::T, b::T, c::T, d::T; check_args=true) where {T <: Real}
    check_args && @check_args(TrapezoidalDist, a <= b <= c <= d)
    return TrapezoidalDist{T}(a, b, c, d)
end


TrapezoidalDist(a::Real, b::Real, c::Real, d::Real) = TrapezoidalDist(promote(a, b, c, d)...)
TrapezoidalDist(a::Integer, b::Integer, c::Integer, d::Integer) = TrapezoidalDist(float(a), float(b), float(c), float(d))

@distr_support TrapezoidalDist d.a d.d

#### Conversions

convert(::Type{TrapezoidalDist{T}}, a::Real, b::Real, c::Real, d::Real) where {T<:Real} = TrapezoidalDist(T(a), T(b), T(c), T(d))
convert(::Type{TrapezoidalDist{T}}, d::TrapezoidalDist{S}) where {T<:Real, S<:Real} = TrapezoidalDist(T(d.a), T(d.b), T(d.c), T(d.d), check_args=false)

#### Parameters

params(d::TrapezoidalDist) = (d.a, d.b, d.c, d.d)
partype(::TrapezoidalDist{T}) where {T<:Real} = T


#### Statistics

function mean(d::TrapezoidalDist)
    ((d.d^3 - d.c^3)/(d.d-d.c) - (d.b^3 - d.a^3)/(d.b-d.a)) / (3*(d.d+d.c-d.b-d.a))
end

function var(d::TrapezoidalDist)
    ((d.d^4 - d.c^4)/(d.d-d.c) - (d.b^4 - d.a^4)/(d.b-d.a)) / (6*(d.d+d.c-d.b-d.a)) - mean(d)^2
end

function entropy(d::TrapezoidalDist{T}) where {T<:Real}
    (a, b, c, d) = params(d)
    (d-c+b-a)/(2*(d+c-b-a)) + log(d+c-b-a) - log(2)
end

#### Evaluation

function pdf(d::TrapezoidalDist{T}, x::Real) where T<:Real
    (a, b, c, d) = params(d)
    x <= a ? zero(T) :
        a <= x <  b ? 2 / (d+c-a-b) * (x-a)/(b-a) :
        b <= x <  c ? 2 / (d+c-a-b) :
        c <= x <= d ? 2 / (d+c-a-b) * (d-a)/(d-c) : zero(T)
end

function cdf(d::TrapezoidalDist{T}, x::Real) where T<:Real
    (a, b, c, d) = params(d)
    x <= a ? zero(T) :
        a <= x <  b ? 1/(d+c-a-b) * (x-a)^2/(b-a) :
        b <= x <  c ? 1/(d+c-a-b) * (2*x-a-b)  :
        c <= x <= d ? 1 - 1/(d+c-a-b) * (d-x)^2/(d-c) : one(T)
end

function mgf(d::TrapezoidalDist{T}, t::Real) where T<:Real
    if t == zero(t)
        return one(T)
    else
        (a, b, c, d) = params(d)
        return 2/(d+c-b-a)/t^2 * ((exp(d*t) - exp(c*t))/(d-c) - (exp(b*t) - exp(a*t))/(b-a))
    end
end


#### Sampling

function rand(rng::AbstractRNG, d::TrapezoidalDist)
    (a, b, c, d) = params(d)
    p_ab = (b-a) / (d+c-a-b)
    p_ac = (2*c-a-b) / (d+c-a-b)
    u = rand(rng)
    u < p_ab ? rand(rng, TriangularDist(a, b, b)) :
        u < p_ac ? rand(rng, Uniform(b, c)) :
        rand(rng, TriangularDist(c, d, c))
end

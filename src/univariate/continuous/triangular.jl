immutable TriangularDist <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64
    c::Float64

    function TriangularDist(a::Real, b::Real, c::Real)
        a < b || error("TriangularDist: a < b must be true")
        a <= c <= b || error("a <= c <= b must be true")
        @compat new(Float64(a), Float64(b), Float64(c))
    end

    function TriangularDist(a::Real, b::Real)
        a < b || error("TriangularDist: a < b must be true")
        @compat a_ = Float64(a)
        @compat b_ = Float64(b)
        c_ = middle(a_, b_)
        new(a_, b_, c_)
    end
end

@distr_support TriangularDist d.a d.b


#### Parameters

params(d::TriangularDist) = (d.a, d.b, d.c)


#### Statistics

mode(d::TriangularDist) = d.c

mean(d::TriangularDist) = (d.a + d.b + d.c) / 3.0

function median(d::TriangularDist)
    (a, b, c) = params(d)
    m = middle(a, b)
    c >= m ? a + sqrt(0.5 * (b - a) * (c - a)) :
             b - sqrt(0.5 * (b - a) * (b - c))
end

_pretvar(a::Float64, b::Float64, c::Float64) = a*a + b*b + c*c - a*b - a*c - b*c

function var(d::TriangularDist)
    (a, b, c) = params(d)
    _pretvar(a, b, c) / 18.0
end

function skewness(d::TriangularDist)
    (a, b, c) = params(d)
    sqrt2 * (a + b - 2.0c) * (2.0a - b - c) * (a - 2.0b + c) / (5.0 * _pretvar(a, b, c)^1.5)
end

kurtosis(d::TriangularDist) = -0.6

entropy(d::TriangularDist) = 0.5 + log((d.b - d.a) / 2.0)


#### Evaluation

function pdf(d::TriangularDist, x::Float64)
    (a, b, c) = params(d)
    x <= a ? 0.0 :
    x <  c ? 2.0 * (x - a) / ((b - a) * (c - a)) :
    x == c ? 2.0 / (b - a) :
    x <= b ? 2.0 * (b - x) / ((b - a) * (b - c)) : 0.0
end

function cdf(d::TriangularDist, x::Float64)
    (a, b, c) = params(d)
    x <= a ? 0.0 :
    x <  c ? (x - a)^2 / ((b - a) * (c - a)) :
    x == c ? (c - a) / (b - a) :
    x <= b ? 1.0 - (b - x)^2 / ((b - a) * (b - c)) : 1.0
end

function quantile(d::TriangularDist, p::Float64)
    (a, b, c) = params(d)
    c_m_a = c - a
    b_m_a = b - a
    rl = c_m_a / b_m_a
    p <= rl ? a + sqrt(b_m_a * c_m_a * p) :
              b - sqrt(b_m_a * (b - c) * (1.0 - p))
end

function mgf(d::TriangularDist, t::Real)
    if t == zero(t)
        return one(t)
    else
        (a, b, c) = params(d)
        u = (b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)
        v = (b - a) * (c - a) * (b - c) * t^2
        return 2.0 * u / v
    end
end

function cf(d::TriangularDist, t::Real)
    # Is this correct?
    if t == zero(t)
        return one(t)
    else
        (a, b, c) = params(d)
        u = (b - c) * cis(a * t) - (b - a) * cis(c * t) + (c - a) * cis(b * t)
        v = (b - a) * (c - a) * (b - c) * t^2
        return -2.0 * u / v
    end
end


#### Sampling

function rand(d::TriangularDist)
    (a, b, c) = params(d)
    b_m_a = b - a
    u = rand()
    b_m_a * u < (c - a) ? d.a + sqrt(u * b_m_a * (c - a)) : 
                          d.b - sqrt((1.0 - u) * b_m_a * (b - c))
end


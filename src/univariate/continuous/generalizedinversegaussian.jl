"""
    GeneralizedInverseGaussian(a, b, p)
The *Generalized Inverse Gaussian distribution* has probability density function 
```math
f(x; a,b,p) = \\frac{(a/b)^{p/2}}{2K_p(\\sqrt{ab})}x^{(p-1)}e^{-(ax + b/x)/2}
x > 0
```

```julia
GeneralizedInverseGaussian(a,b,p)   # GeneralizedInverseGaussian distribution with parametrs a, b, and p

params(d)                           # Get the parameters, i.e. (a, b, p)
```

External Links

* [generalized inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution)
* [Sampling implementation paper](https://doi.org/10.1007/s11222-013-9387-3)
"""
struct GeneralizedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    p::T
    GeneralizedInverseGaussian{T}(a::T, b::T, p::T) where T = new{T}(a,b,p)
end

function GeneralizedInverseGaussian(a::T,b::T,p::T,check_args=true) where {T<:Real}
    check_args && @check_args(GeneralizedInverseGaussian, a > zero(a) && b > zero(b))
    return GeneralizedInverseGaussian{T}(a,b,p)
end

GeneralizedInverseGaussian(a::Real, b::Real, p::Real) = GeneralizedInverseGaussian(promote(a,b,p)...)
GeneralizedInverseGaussian(a::Integer, b::Integer, p::Integer) = GeneralizedInverseGaussian(float(a), float(b), float(p))

@distr_support GeneralizedInverseGaussian 0.0 Inf

#### Conversions
function convert(::Type{GeneralizedInverseGaussian{T}}, a::Real, b::Real, p::Real) where T<:Real
    GeneralizedInverseGaussian(T(a),T(b),T(p))
end

function convert(::Type{GeneralizedInverseGaussian{T}}, d::GeneralizedInverseGaussian{S}) where {T <: Real, S<: Real}
    GeneralizedInverseGaussian(T(d.a), T(d.b), T(d.p))    
end

#### Parameters

params(d::GeneralizedInverseGaussian) = (d.a, d.b, d.p)
@inline partype(d::GeneralizedInverseGaussian{T}) where {T<:Real} = T

#### Statistics

mean(d::GeneralizedInverseGaussian) = ((a,b,p) = params(d); (sqrt(b) * besselk(p+1,sqrt(a*b))) / (sqrt(a) * besselk(p,sqrt(a*b))))

mode(d::GeneralizedInverseGaussian) = ((a,b,p) = params(d); ((p - 1) + sqrt((p - 1)^2 + a*b) / a))

function var(d::GeneralizedInverseGaussian) 
    (a,b,p) = params(d)
    left = besselk(p+2,sqrt(a*b)) / besselk(p, sqrt(a*b))
    right = besselk(p+1,sqrt(a*b)) / besselk(p, sqrt(a*b))
    return (b/a) * (left - right^2)
end

#### Evaluation

function pdf(d::GeneralizedInverseGaussian{T}, x::Real) where {T<:Real}
    if x > 0
        (a,b,p) = params(d)
        top = (a/b)^(p/2)
        bot = 2*besselk(p,sqrt(a*b))
        right = x^(p-1) * exp(-(a*x + (b/x))/2)
        return (top/bot)*right
    end
    return zero(T)
end

cdf(d::GeneralizedInverseGaussian{T}, x::Real) where {T<:Real} = throw(MethodError(cdf, (d, x)))


function logpdf(d::GeneralizedInverseGaussian{T}, x::Real) where {T<:Real}
    if (x > 0)
        (a,b,p) = params(d)
        top = (p/2)*(log(a) - log(b))
        bot = log(2*besselk(p,sqrt(a*b)))
        left = (p - 1)*log(x)
        right = -(a*x + (b/x))/2
        return top - bot + left - right
    end
    return -T(Inf)
end

function mgf(d::GeneralizedInverseGaussian{T}, t::Real) where {T<:Real}
    (a,b,p) = params(d)
    left = (a / (a - 2*t))^(p/2)
    top = besselk(p,sqrt(b*(a - 2*t)))
    bot = besselk(p,sqrt(a*b))
    return left * (top/bot)
end

function cf(d::GeneralizedInverseGaussian{T}, t::Real) where {T<:Real}
    (a,b,p) = params(d)
    left = (a / (a - 2*im*t))^(p/2)
    top = besselk(p,sqrt(b * (a - 2*im*t)))
    bot = besselk(p,sqrt(a*b))
    return left * (top/bot)
end

#### Sampling 

"""
    rand(d)

Extract a sample from the GeneralizedInverseGaussian distribution 'd'. 
The sampling procedure is implemented from [1].
One of the following conditions must be satisfied:

1. 0 < a*b < min(1/2, (2/3)*sqrt(1-p)) and 0 ≤ p < 1
2. min(1/2, (2/3)*sqrt(1-p)) ≤ a*b ≤ 1 and 0 ≤ p < 1
3. β > 1 and p > 1

[1] Hörmann, W., Leydold, J. (2014).
Generating generalized inverse Gaussian random variates. 
Stat Comput 24, 547–557. 
https://doi.org/10.1007/s11222-013-9387-3
"""
function rand(d::GeneralizedInverseGaussian)
    (a,b,p) = params(d)
    β = sqrt(a*b)
    # TODO: better ifs here so as not to check the "p"s twice
    β_bound = min(1/2, (2/3)*sqrt(1 - p))
    if(β < β_bound && β > 0 && p >= 0 && p < 1) 
        return concave_sample(p,β)
    elseif(β >= β_bound && β <= 1 && p >= 0 && p < 1)
        return sample_unif_no_mode_shift(p,β)
    elseif(β > 1 && p > 1)
        return sample_unif_mode_shift(p,β)
    else
        throw(ArgumentError("None of the required conditions on the parameters are satisfied"))
    end
end

function concave_sample(p::Real,β::Real)
    m = β/((1 - p) + sqrt((1 - p)^2 + β^2))
    x_naut = β/(1 - p)
    x_star = max(x_naut,2/β)
    k1 = g(m,p,β)
    A1 = k1 * x_naut
    k2 = 0
    A2 = 0
    if (x_naut < 2/β)
        k2 = exp(-β)
        if (p > 0)
            A2 = k2 * ((2/β)^p - x_naut^p) / p
        else
            A2 = k2 * log(2/β^2)
        end
    end
    k3 = x_star^(p-1)
    A3 = 2 * k3 * exp(-x_star * β / 2) / β
    A = A1 + A2 + A3

    while true
        u = rand(Uniform(0,1))
        v = rand(Uniform(0,A))
        h = Inf
        if(v <= A1)
            x = x_naut * v / A1
            h = k1
        elseif(v <= A1 + A2)
            v = v - A1
            if (p > 0)
                x = (x_naut^p + (v*p/k2))^(1/p)
            else
                x = β*exp(v * exp(β))
            end
            h = k2 * x^(p-1)
        else
            v = v - (A1 + A2)
            x = -2 * log(exp(-x_star * β / 2) - (v * β) / (2 * k3)) / β
            h = k3 * exp(-x * β / 2)
        end
        if(u * h <= g(x,p,β))
            return x
        end
    end
end

function sample_unif_no_mode_shift(p::Real,β::Real)
    m = β/((1 - p) + sqrt((1 - p)^2 + β^2))
    x⁺= ((1 + p) + sqrt((1 + p)^2 + β^2))/β
    v⁺= sqrt(g(m,p,β))
    u⁺= x⁺ * sqrt(g(x⁺,p,β))
    while true
        u = rand(Uniform(0,u⁺))
        v = rand(Uniform(0,v⁺))
        x = u/v
        if (v^2 <= g(x,p,β))
            return x
        end
    end
end

function sample_unif_mode_shift(p::Real,β::Real)
    m = (sqrt((p - 1)^2 + β^2) + (p-1)) / β
    a = -(2*(p+1)/β) - m
    b = -(2*(p-1)/β) * m - 1
    c = m
    p2 = b - (a^2)/3
    q = (2*a^3)/27 - (a*b/3) + c
    ϕ = arccos(-(q/2) * sqrt(-27 / (p2^3)))
    x⁻ = sqrt((-4/3)*p2)*cos(ϕ/3 + (4/3)π) - (a / 3)
    x⁺ = sqrt((-4/3)*p2)*cos(ϕ/3) - (a/3)
    v⁺ = sqrt(g(m,p,β))
    u⁻ = (x⁻ - m)*sqrt(g(x⁻,p,β))
    u⁺ = (x⁺ - m)*sqrt(g(x⁺,p,β))

    while true
        u = rand(Uniform(u⁻,u⁺))
        v = rand(Uniform(0,v⁺))
        x = (u / v) + m
        if(v^2 <= g(x,p,β))
            return x
        end
    end
end

function g(x::Real,p::Real,β::Real)
    x^(p-1)*exp(-(β/2)*(x + (1/x)))
end

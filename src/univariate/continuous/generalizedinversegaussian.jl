doc"""
    GeneralizedInverseGaussian(a,b,p)

The *generalized inverse Gaussian distribution* with parameters a > 0, b > 0, p real,
and modified Bessel function of the second kind K_p, has probability density function

$f(x; a, b, p) = \frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})}x^{(p-1)}
e^{-(ax+b/x)/2}, \quad x > 0$

``julia
GeneralizedInverseGaussian(a, b, p)    # Generalized Inverse Gaussian distribution with parameters parameters a > 0, b > 0 and p real

params(d)           # Get the parameters, i.e. (a, b, p)
``

External links

* [Generalized Inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution)

"""
immutable GeneralizedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    p::T

    function GeneralizedInverseGaussian(a::T, b::T, p::T)
        @check_args(GeneralizedInverseGaussian, a > zero(a) && b > zero(b)) 
        new(a, b, p)
    end
end

GeneralizedInverseGaussian{T<:Real}(a::T, b::T, p::T) = GeneralizedInverseGaussian{T}(a, b, p)
GeneralizedInverseGaussian(a::Real, b::Real, p::Real) = GeneralizedInverseGaussian(promote(a, b, p)...)
GeneralizedInverseGaussian(a::Integer, b::Integer, p::Integer) = GeneralizedInverseGaussian(Float64(a), Float64(b), Float64(p))

@distr_support GeneralizedInverseGaussian 0.0 Inf


#### Conversions

function convert{T <: Real, S <: Real}(::Type{GeneralizedInverseGaussian{T}}, a::S, b::S, p::S)
    GeneralizedInverseGaussian(T(a), T(b), T(p))
end
function convert{T <: Real, S <: Real}(::Type{GeneralizedInverseGaussian{T}}, d::GeneralizedInverseGaussian{S})
    GeneralizedInverseGaussian(T(d.a), T(d.b), T(d.p))
end

#### Parameters

params(d::GeneralizedInverseGaussian) = (d.a, d.b, d.p)
@inline partype{T<:Real}(d::GeneralizedInverseGaussian{T}) = T


#### Statistics

function mean(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    q = sqrt(a * b)
    (sqrt(b) * besselk(p + 1, q)) / (sqrt(a) * besselk(p, q))
end

function var(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    q = sqrt(a * b)
    r = besselk(p, q)
    (b / a) * ((besselk(p + 2, q) / r) - (besselk(p + 1, q) / r)^2)
end

mode(d::GeneralizedInverseGaussian) = ((d.p - 1) + sqrt((d.p - 1)^2 + d.a * d.b)) / d.a


#### Evaluation

function pdf{T<:Real}(d::GeneralizedInverseGaussian{T}, x::Real)
    if x > 0
        a, b, p = params(d)
        (((a / b)^(p / 2)) / (2 * besselk(p, sqrt(a * b)))) * (x^(p - 1)) * exp(- (a * x + b / x) / 2)
    else
        zero(T)
    end
end

function logpdf{T<:Real}(d::GeneralizedInverseGaussian{T}, x::Real)
    if x > 0
        a, b, p = params(d)
        (p / 2) * (log(a) - log(b)) - log(2 * besselk(p, sqrt(a * b))) + (p - 1) * log(x) - (a * x + b / x) / 2
    else
        -T(Inf)
    end
end


function cdf{T<:Real}(d::GeneralizedInverseGaussian{T}, x::Real)
    if x > 0
        # See eq. (5) in Lemonte & Cordeiro (2011) 
        # Statistics & Probability Letters 81:506–517
        # F(x) = 1 - (ρ + σ), where ρ and σ are infinite sums
        # calculated up to truncation below
        a, b, p = params(d)
        c = (((a / b)^(p / 2)) / (2 * besselk(p, sqrt(a * b))))
        η = a / 2
        ω = b / 2
        lη = log(η)
        lω = log(ω)
        lx = log(x)
        # calculate first term ρ
        ρ = 0.0
        converged = false
        j = 0
        while !converged && j < 100
            ρ_old = ρ
            ρ += c * (-1)^j * gamma(p - j) * exp((-p + j) * lη + j * lω - lfact(j))
            converged = abs(ρ - ρ_old) < eps()
            j += 1
        end
        # calculate second term σ
        σ = 0.0
        converged = false
        i = 0
        while !converged && i < 100
            σ_old = σ
            j = 0
            k = i
            while j <= i
                l = k * lη + j * lω + (k - j + p) * lx - lfact(k) - lfact(j) 
                σ += (c * (-1)^(k + j + 1) * exp(l)) / (k - j + p)
                j += 1
                k -= 1
            end
            converged = abs(σ - σ_old) < eps()
            i += 1
        end
        1 - (ρ + σ)
    else
        zero(T)
    end
end

function mgf{T <: Real}(d::GeneralizedInverseGaussian{T}, t::Real)
    if t == zero(t)
        one(T)
    else
        a, b, p = params(d)
        (a / (a - 2t))^(p / 2) * besselk(p, sqrt(b * (a - 2t))) / besselk(p, sqrt(a * b))
    end
end

function cf{T <: Real}(d::GeneralizedInverseGaussian{T}, t::Real)
    if t == zero(t)
        one(T) + zero(T) * im
    else
        a, b, p = params(d)
        (a / (a - 2t * im))^(p / 2) * besselk(p, sqrt(b * (a - 2t * im))) / besselk(p, sqrt(a * b))
    end
end



#### Sampling

# rand method from:
# Hörmann, W. & J. Leydold. (2014). Generating generalized inverse Gaussian random variates.
# J. Stat. Comput. 24: 547–557. doi:10.1007/s11222-013-9387-3

function rand(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    α = sqrt(a / b)
    β = sqrt(a * b)
    λ = abs(p)
    if (λ > 1) || (β > 1)
        x = _rou_shift(λ, β)
    elseif β >= min(0.5, (2 / 3) * sqrt(1 - λ))
        x = _rou(λ, β)
    else
        x = _hormann(λ, β)
    end
    p >= 0 ? x / α : 1 / (α * x)
end

function _gigqdf(x::Real, λ::Real, β::Real)
    (x^(λ - 1)) * exp(-β * (x + 1 / x) / 2) 
end

function _hormann(λ::Real, β::Real)
    # compute bounding rectangles
    m = β / (1 - λ + sqrt((1 - λ)^2 + β^2))  # mode
    x0 = β / (1 - λ)
    xstar = max(x0, 2 / β)
    # in subdomain (0, x0)
    k1 = _gigqdf(m, λ, β)
    a1 = k1 * x0
    # in subdomain (x0, 2 / β), may be empty
    if x0 < 2 / β
        k2 = exp(-β)
        a2 = λ == 0 ? k2 * log(2 / (β^2)) : k2 * ((2 / β)^λ - x0^λ) / λ
    else
        k2 = 0
        a2 = 0
    end
    # in subdomain (xstar, Inf)
    k3 = xstar^(λ - 1)
    a3 = 2k3 * exp(-xstar * β / 2) / β
    a = a1 + a2 + a3

    # perform rejection sampling
    while true
        u = rand()
        v = a * rand()
        if v <= a1
            # in subdomain (0, x0)
            x = x0 * v / a1
            h = k1
        elseif v <= a1 + a2
            # in subdomain (x0, 2 / β)
            v -= a1
            x = λ == 0 ? β * exp(v * exp(β)) : (x0^λ + v * λ / k2)^(1 / λ)
            h = k2 * x^(λ - 1)
        else
            # in subdomain (xstar, Inf)
            v -= a1 + a2
            x = -2log(exp(-xstar * β / 2) - v * β / (2k3)) / β
            h = k3 * exp(-x * β / 2)
        end
        if u * h <= _gigqdf(x, λ, β)
            return x
        end
    end
end

function _rou(λ::Real, β::Real)
    # compute bounding rectangle
    m = β / (1 - λ + sqrt((1 - λ)^2 + β^2))  # mode
    xpos = (1 + λ + sqrt((1 + λ)^2 + β^2)) / β
    vpos = sqrt(_gigqdf(m, λ, β))
    upos = xpos * sqrt(_gigqdf(xpos, λ, β))
    
    # perform rejection sampling
    while true
        u = upos * rand()
        v = vpos * rand()
        x = u / v
        if v^2 <= _gigqdf(x, λ, β)
            return x
        end
    end
end

function _rou_shift(λ::Real, β::Real)
    # compute bounding rectangle
    m = (λ - 1 + sqrt((λ - 1)^2 + β^2)) / β  # mode
    a = -2(λ + 1) / β - m
    b = 2(λ - 1) * m / β - 1
    p = b - (a^2) / 3
    q = 2(a^3) / 27 - (a * b) / 3 + m
    ϕ = acos(-(q / 2) * sqrt(-27 / (p^3)))  # Cardano's formula
    r = sqrt(-4p / 3)
    xneg = r * cos(ϕ / 3 + 4π / 3) - a / 3
    xpos = r * cos(ϕ / 3) - a / 3
    vpos = sqrt(_gigqdf(m, λ, β))
    uneg = (xneg - m) * sqrt(_gigqdf(xneg, λ, β))
    upos = (xpos - m) * sqrt(_gigqdf(xpos, λ, β))

    # perform rejection sampling
    while true
        u = (upos - uneg) * rand() + uneg
        v = vpos * rand()
        x = max(u / v + m, 0)
        if v^2 <= _gigqdf(x, λ, β)
            return x
        end
    end
end


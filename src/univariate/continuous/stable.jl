# methods mainly from John P. Nolan, "Univariate Stable Distributions", Springer 2020
"""
    Stable(α, β, σ, μ)

The *Stable distribution* with stability index 0 < `α` ≤ 2, skewness parameter -1 ≤ `β` ≤ 1, scale 0 < `σ` and location `μ` has characteristic function

```math
\\varphi(t; \\alpha, \\beta, \\sigma, \\mu) = \\exp\\big(\\mathrm i t\\mu -|\\sigma t|^\\alpha(1-\\mathrm i\\beta\\mathrm{sgn}(t)\\Phi(t))\\big)
```
with ``\\Phi(t) = -\\frac{2}{\\pi}\\log|t|`` for α = 1 or ``\\Phi(t) = \\tan(\\pi\\alpha/2)`` for α ≠ 1. We use type-1 parametrization.


```julia
Stable(α)             # standard symmetric α-Stable distribution equivalent to Stable(α, 0, 1, 0)
Stable(α, β)          # standard α-Stable distribution with skewness parameter β equivalent to S(α, β, 1, 0)
Stable(α, β, σ, μ)    # α-Stable distribution with skewness parameter β, scale σ and location μ

params(d, [type=:type1])      # Get the parameters, i.e. (α, β, σ, μ) in a chosen parametrization
shape(d)                      # Get the shape, i.e. (α, β)
location(d)                   # Get the location, i.e. μ
scale(d)                      # Get the scale, i.e. σ
minimum(d)                    # Get the lower bound
maximum(d)                    # Get the upper bound
```

External links

* [Stable distribution on Wikipedia](https://en.wikipedia.org/wiki/Stable_distribution)

"""
struct Stable{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    σ::T
    μ::T
    Stable{T}(α, β, σ, μ) where {T} = new{T}(α, β, σ, μ)
end

function Stable(α::T, β::T, σ::T, μ::T; check_args::Bool=true) where {T <: Real}
    @check_args Stable (α, zero(T) < α <= 2one(T)) (β, -one(T) <= β <= one(T)) (σ, zero(T) < σ) ((α, β), α != 2 || β == 0 )
    return Stable{T}(α, β, σ, μ)
end

Stable(α::Real, β::Real, σ::Real, μ::Real; check_args::Bool=true) = Stable(promote(α, β, σ, μ)...; check_args=check_args)
Stable(α::Integer, β::Integer, σ::Integer, μ::Integer; check_args::Bool=true) = Stable(float(α), float(β), float(σ), float(μ); check_args=check_args)
Stable(α::Real, β::Real; check_args::Bool=true) = Stable(α, β, one(α), zero(α); check_args=check_args)
Stable(α::Real; check_args::Bool=true) = Stable(α,zero(α); check_args=check_args)

@distr_support Stable (d.α < 1 && d.β == 1 ? d.μ : -Inf) (d.α < 1 && d.β == -1 ? d.μ : Inf)

#### Conversions

convert(::Type{Stable{T}}, α::S, β::S, σ::S, μ::S) where {T <: Real, S <: Real} = Stable(T(α), T(β), T(σ), T(μ))
Base.convert(::Type{Stable{T}}, d::Stable) where {T<:Real} = Stable{T}(T(d.α), T(d.β), T(d.σ), T(d.μ))
Base.convert(::Type{Stable{T}}, d::Stable{T}) where {T<:Real} = d

#### Parameters

shape(d::Stable) = (d.α, d.β)
location(d::Stable) = d.μ
scale(d::Stable) = d.σ

"""
    params(d::Stable, [type::Symbol=:type1])

Return a tuple of parameters in a chosen parametrization. Type-0 and type-1 (default) are implemented.

"""
params(d::Stable{T}, type::Symbol = :type1) where T = 
    if type ∈ (:type1, :t1, :one)
        return (d.α, d.β, d.σ, d.μ)
    elseif type ∈ (:type0, :t0, :zero)
        return (d.α, d.β, d.σ, d.μ + d.β*d.σ*(d.α == one(T) ? 2/π*log(d.σ) : tan(d.α*π/2)))
    else # type-2 may be added, see "Univariate Stable Distributions" J.P. Nolan
        throw(ArgumentError("Unrecognized parametrization type. Choose between :type0, :t0, :zero or  :type1, :t1, :one."))
    end
partype(::Stable{T}) where {T} = T

#### Statistics

mean(d::Stable{T}) where T = d.α > one(T) ? d.μ : T(NaN)
var(d::Stable{T}) where T = d.α == 2one(T) ? 2d.σ^2 : T(Inf)
skewness(d::Stable{T}) where T = d.α == 2one(T) ? T(0.0) : T(NaN)
kurtosis(d::Stable{T}) where T = d.α == 2one(T) ? T(3.0) : T(NaN)

#### Evaluation

# integral representation from Nolan ch. 3
function pdf(d::Stable{T}, x::Real) where T
    α, β, σ, μ =  params(d)

    α == 2one(T) && return pdf(Normal(μ, √2σ),x)
    β == zero(T) && return pdf(Cauchy(μ,σ),x)
    α == one(T)/2 && β == one(T) && return pdf(Levy(μ, σ), x)
    α == one(T)/2 && β == -one(T) && return pdf(Levy(-μ, σ), -x)

    w(v,c) = v*c > 36. ? 0.0 : v*exp(-c*v) # numerical truncation

    if α == one(T) 
        V₁(θ) = 2/π*(π/2+β*θ)/cos(θ) * exp((π/2+β*θ)*tan(θ)/β)

        x = (x-μ)/σ - 2/π*β*log(σ) # normalize to S(1,β,1,0)
        x < 0 && ( (x, β, μ) = (-x, -β, -μ) ) # reflection property

        I, _err = quadgk(θ -> w(V₁(θ),exp(-π*x/2β)), -π/2, π/2 ) 

        return 1/(2abs(β)*σ) * exp(-π*x/2β) * I
    else 
        V(θ) =(cos(α*θ₀))^(1/(α-1)) * (cos(θ)/sin(α*(θ₀+θ)))^(α/(α-1)) * cos(α*θ₀ + (α-1)*θ)/cos(θ)

        x = (x-μ)/σ # normalize to S(α,β,1,0)
        x < 0 && ( (x, β, μ) = (-x, -β, -μ) ) # reflection property

        θ₀ =  atan(β*tan(α*π/2))/α
        x ≈ 0. && return gamma(1+1/α)*cos(θ₀)*(cos(α*θ₀))^(1/α) / π
        
        I, _err =  quadgk(θ -> w(V(θ), x^(α/(α-1)) ), -θ₀, π/2)

        return α/σ * x^(1/(α-1)) / (π*abs(α-1)) * I
    end
end

function logpdf(d::Stable{T}, x::Real) where T
    α, β, σ, μ = params(d)
    α == 2one(T) && return logpdf(Normal(μ, √2σ),x)
    α == one(T) && β == zero(T) && return logpdf(Cauchy(μ,σ),x)
    α == one(T)/2 && β == one(T) && return logpdf(Levy(μ, σ), x)
    α == one(T)/2 && β == -one(T) && return logpdf(Levy(-μ, σ), -x)
    return log(pdf(d,x))
end

# integral representation from Nolan ch. 3
function cdf(d::Stable{T}, x::Real) where T
    α, β, σ, μ =  params(d)

    α == 2one(T) && return cdf(Normal(μ, √2σ),x)
    β == zero(T) && return cdf(Cauchy(μ,σ),x)
    α == one(T)/2 && β == one(T) && return cdf(Levy(μ, σ), x)
    α == one(T)/2 && β == -one(T) && return 1 - cdf(Levy(-μ, σ), -x)

    z(v,c) = v*c > 36. ? 0.0 : exp(-c*v) # numerical truncation

    function F(α,β,x) # works for x > 0
        V(θ) =(cos(α*θ₀))^(1/(α-1)) * (cos(θ)/sin(α*(θ₀+θ)))^(α/(α-1)) * cos(α*θ₀ + (α-1)*θ)/cos(θ)
        θ₀=  atan(β*tan(α*π/2))/α
        x ≈ 0. && return (π/2 - θ₀)/π

        c = α > 1 ? 1. : (π/2 - θ₀)/π
        I, _err =  quadgk(θ -> z(V(θ), x^(α/(α-1)) ), -θ₀, π/2)
        return c + sign(1-α)/π * I
    end

    function F₁(β,x) # works for β > 0
        V₁(θ) = 2/π*(π/2+β*θ)/cos(θ) * exp((π/2+β*θ)*tan(θ)/β)
        I, _err = quadgk(θ -> z(V₁(θ),exp(-π*x/2β)), -π/2, π/2 ) 
        return 1/π* I
    end

    if α == one(T) 
        x = (x-μ)/σ - 2/π*β*log(σ) # normalize to S(1,β,1,0)
        β < 0 && return F₁(-β,x)
        return F₁(β,x)
    else
        x = (x-μ)/σ # normalize to S(α,β,1,0)
        x < 0 && return 1 - F(α,-β,-x)
        return F(α,β,x)
        
    end
end

@quantile_newton Stable

#### Affine transformations

Base.:+(d::Stable, a::Real) = Stable(d.α, d.β, d.σ, d.μ + a)
Base.:*(c::Real, d::Stable{T}) where T = 
    if d.α == one(T)
        Stable(d.α, sign(c)*d.β, abs(c)*d.σ, c*d.μ - 2/π*d.β*d.σ*c*log(abs(c)))
    else
        Stable(d.α, sign(c)*d.β, abs(c)*d.σ, c*d.μ)
    end

#### Sampling

# A. Weron, R. Weron "Computer simulation of Lévy α-stable variables and processes", Springer 1995, doi: 10.1007/3-540-60188-0_6
function rand(rng::AbstractRNG, d::Stable{T}) where T
    (α, β, σ, μ) = params(d)

    α == 2one(T) && return rand(rng, Normal(μ, √2σ)) # Gaussian case
    α == one(T) && β == zero(T) && return rand(rng, Cauchy(μ, σ)) # Cauchy case
    α == one(T)/2 && β == one(T) && return rand(rng,Levy(μ, σ)) # Lévy case
    α == one(T)/2 && β == -one(T) && return -rand(rng,Levy(-μ, σ))

    v = π*rand(rng) - π/2
    w = randexp(rng)

    β == zero(T) && return σ  * ( sin(α*v)/cos(v)^(1/α) * (cos((1-α)*v)/w)^((1-α)/α) ) + μ # symmetric stable
    α == one(T) && return σ * 2/π*( (π/2 + β*v)*tan(v) - β*log( (π/2*w*cos(v)/(π/2 + β*v)) ) ) + 2/π*β*σ*log(σ) + μ # 1-stable

    b = atan(β*tan(π*α/2))/α
    return σ * sin(α*(v+b))/(cos(α*b) * cos(v))^(1/α) * (cos((1-α)*v - α*b)/w)^((1-α)/α) + μ # general case
end

#### Fit model

function _crop(x::Real,a::Real,b::Real)
    x = max(x, a)
    x = min(x, b)
end 

# McCulloch's quantile method
function fit_quantile(::Type{<:Stable}, x::AbstractArray{<:Real})
    function ψ₁(x::Real,y::Real)
        i = [2.439, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0]
        j = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] 
        x = _crop(x, i[1], i[end])

        Ψ₁=[2.000 2.000 2.000 2.000 2.000 2.000 2.000;
            1.916 1.924 1.924 1.924 1.924 1.924 1.924;
            1.808 1.813 1.829 1.829 1.829 1.829 1.829;
            1.729 1.730 1.737 1.745 1.745 1.745 1.745;
            1.664 1.663 1.663 1.668 1.676 1.676 1.676;
            1.563 1.560 1.553 1.548 1.547 1.547 1.547;
            1.484 1.480 1.471 1.460 1.448 1.438 1.438;
            1.391 1.386 1.378 1.364 1.337 1.318 1.318;
            1.279 1.273 1.266 1.250 1.210 1.184 1.150;
            1.128 1.121 1.114 1.101 1.067 1.027 0.973;
            1.029 1.021 1.014 1.004 0.974 0.935 0.874;
            0.896 0.892 0.887 0.883 0.855 0.823 0.769;
            0.818 0.812 0.806 0.801 0.780 0.756 0.691;
            0.698 0.695 0.692 0.689 0.676 0.656 0.595;
            0.593 0.590 0.588 0.586 0.579 0.563 0.513]
         itp = interpolate((i, j), Ψ₁, Gridded(Linear()))
         return itp(x, abs(y))
    end
    
    function ψ₂(x::Real,y::Real)
        i = [2.439, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0]
        j = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] 
        x = _crop(x, i[1], i[end])
        
        Ψ₂=[0.000 2.160 1.000 1.000 1.000 1.000 1.000;
            0.000 1.592 3.390 1.000 1.000 1.000 1.000;
            0.000 0.759 1.800 1.000 1.000 1.000 1.000;
            0.000 0.482 1.048 1.694 1.000 1.000 1.000;
            0.000 0.360 0.760 1.232 2.229 1.000 1.000;
            0.000 0.253 0.518 0.823 1.575 1.000 1.000;
            0.000 0.203 0.410 0.632 1.244 1.906 1.000;
            0.000 0.165 0.332 0.499 0.943 1.560 1.000;
            0.000 0.136 0.271 0.404 0.689 1.230 2.195;
            0.000 0.109 0.216 0.323 0.539 0.827 1.917;
            0.000 0.096 0.190 0.284 0.472 0.693 1.759;
            0.000 0.082 0.163 0.243 0.412 0.601 1.596;
            0.000 0.074 0.147 0.220 0.377 0.546 1.482;
            0.000 0.064 0.128 0.191 0.330 0.478 1.362;
            0.000 0.056 0.112 0.167 0.285 0.428 1.274]
         itp = interpolate((i, j), Ψ₂, Gridded(Linear()))
         return sign(y) * itp(x, abs(y))
    end
    
    function ψ₃(x::Real,y::Real)
        i = 0.5:0.1:2.0 
        j = 0.0:0.25:1.0
        x = _crop(x, i[1], i[end])

        Ψ₃=[2.588 3.073 4.534 6.636 9.144;
            2.337 2.635 3.542 4.808 6.247;
            2.189 2.392 3.004 3.844 4.775;
            2.098 2.244 2.676 3.265 3.912;
            2.040 2.149 2.461 2.886 3.356;
            2.000 2.085 2.311 2.624 2.973;
            1.980 2.040 2.205 2.435 2.696;
            1.965 2.007 2.125 2.294 2.491;
            1.955 1.984 2.067 2.188 2.333;
            1.946 1.967 2.022 2.106 2.211;
            1.939 1.952 1.988 2.045 2.116;
            1.933 1.940 1.962 1.997 2.043;
            1.927 1.930 1.943 1.961 1.987;
            1.921 1.922 1.927 1.936 1.947;
            1.914 1.915 1.916 1.918 1.921;
            1.908 1.908 1.908 1.908 1.908]
         itp = interpolate((i, j), Ψ₃, Gridded(Linear()))
         return itp(x, abs(y))
    end

    function ψ₄(x::Real,y::Real) # this is ψ₅ in original McCuloch paper
        i = 0.5:0.1:2.0 
        j = 0.0:0.25:1.0 
        x = _crop(x, i[1], i[end])

        Ψ₄=[0.0 -0.061 -0.279 -0.659 -1.198;
            0.0 -0.078 -0.272 -0.581 -0.997;
            0.0 -0.089 -0.262 -0.520 -0.853;
            0.0 -0.096 -0.250 -0.469 -0.742;
            0.0 -0.099 -0.237 -0.424 -0.652;
            0.0 -0.098 -0.223 -0.383 -0.576;
            0.0 -0.095 -0.208 -0.346 -0.508;
            0.0 -0.090 -0.192 -0.310 -0.447;
            0.0 -0.084 -0.173 -0.276 -0.390;
            0.0 -0.075 -0.154 -0.241 -0.335;
            0.0 -0.066 -0.134 -0.206 -0.283;
            0.0 -0.056 -0.111 -0.170 -0.232;
            0.0 -0.043 -0.088 -0.132 -0.179;
            0.0 -0.030 -0.061 -0.092 -0.123;
            0.0 -0.017 -0.032 -0.049 -0.064;
            0.0  0.000  0.000  0.000  0.000]
         itp = interpolate((i, j), Ψ₄, Gridded(Linear()))
         return sign(y)*itp(x,abs(y))
    end
    
    q₉₅, q₇₅, q₅₀, q₂₅, q₀₅ = quantile(x,(0.95,0.75,0.50,0.25,0.05))

    v₁ = (q₉₅ - q₀₅) / (q₇₅ - q₂₅)
    v₂ = (q₉₅ + q₀₅ - 2q₅₀) / (q₉₅ - q₀₅)

    αₑₛₜ = ψ₁(v₁, v₂)
    βₑₛₜ = ψ₂(v₁, v₂)
    βₑₛₜ = _crop(βₑₛₜ, -1., 1.)
    σₑₛₜ = (q₇₅ - q₂₅) / ψ₃(αₑₛₜ, βₑₛₜ)
    μₑₛₜ =
        if 0.9 < αₑₛₜ < 1.1 
            q₅₀ + σₑₛₜ * ψ₄(αₑₛₜ, βₑₛₜ)
        else
            q₅₀ + σₑₛₜ * (ψ₄(αₑₛₜ, βₑₛₜ) - βₑₛₜ*tan(αₑₛₜ*π/2))
        end

    return Stable(αₑₛₜ, βₑₛₜ, σₑₛₜ, μₑₛₜ)
end

# ECF method, see Nolan ch. 4.
# quantie_fit is used for a robust initial estimate
function fit(::Type{<:Stable}, x::AbstractArray{<:Real}) 
    α₀, _β₀, σ₀, μ₀ = params(fit_quantile(Stable, x))
    u = exp.(LinRange(α₀ > 1 ? -2.0 : 10α₀-12, 0, 10))
    ecf = [mean(cis(t * (val-μ₀)/σ₀) for val in x) for t in u]
    r, θ = abs.(ecf), angle.(ecf)

    mU, mR = [ones(10) -log.(u)], -log.(-log.(r))
    b, αₑₛₜ = (mU'*mU)\(mU'*mR) # ols regression

    αₑₛₜ = _crop(αₑₛₜ, 0., 2.)
    σ₁ = exp(b/αₑₛₜ)

    η(u) = - tan(αₑₛₜ*π/2)*sign(u)*abs(σ₁*u)^αₑₛₜ
    mΘ = [η.(u) u]
    c, μ₁ =  (mΘ'*mΘ)\(mΘ'*θ) # ols regression

    βₑₛₜ = -c/exp(b)
    βₑₛₜ = _crop(βₑₛₜ , -1., 1.)
    σₑₛₜ = σ₀ * σ₁
    μₑₛₜ = σ₀*μ₁ + μ₀

    return Stable(αₑₛₜ, βₑₛₜ, σₑₛₜ, μₑₛₜ)
end
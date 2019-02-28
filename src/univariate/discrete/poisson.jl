"""
    Poisson(λ)

A *Poisson distribution* descibes the number of independent events occurring within a unit time interval, given the average rate of occurrence `λ`.

```math
P(X = k) = \\frac{\\lambda^k}{k!} e^{-\\lambda}, \\quad \\text{ for } k = 0,1,2,\\ldots.
```

```julia
Poisson()        # Poisson distribution with rate parameter 1
Poisson(lambda)       # Poisson distribution with rate parameter lambda

params(d)        # Get the parameters, i.e. (λ,)
mean(d)          # Get the mean arrival rate, i.e. λ
```

External links:

* [Poisson distribution on Wikipedia](http://en.wikipedia.org/wiki/Poisson_distribution)

"""
struct Poisson{T<:Real} <: DiscreteUnivariateDistribution
    λ::T

    Poisson{T}(λ::Real) where {T} = (@check_args(Poisson, λ >= zero(λ)); new{T}(λ))
end

Poisson(λ::T) where {T<:Real} = Poisson{T}(λ)
Poisson(λ::Integer) = Poisson(Float64(λ))
Poisson() = Poisson(1.0)

@distr_support Poisson 0 (d.λ == zero(typeof(d.λ)) ? 0 : Inf)

#### Conversions
convert(::Type{Poisson{T}}, λ::S) where {T <: Real, S <: Real} = Poisson(T(λ))
convert(::Type{Poisson{T}}, d::Poisson{S}) where {T <: Real, S <: Real} = Poisson(T(d.λ))

### Parameters

params(d::Poisson) = (d.λ,)
@inline partype(d::Poisson{T}) where {T<:Real} = T

rate(d::Poisson) = d.λ


### Statistics

mean(d::Poisson) = d.λ

mode(d::Poisson) = floor(Int,d.λ)

function modes(d::Poisson)
    λ = d.λ
    isinteger(λ) ? [round(Int, λ) - 1, round(Int, λ)] : [floor(Int, λ)]
end

var(d::Poisson) = d.λ

skewness(d::Poisson) = one(typeof(d.λ)) / sqrt(d.λ)

kurtosis(d::Poisson) = one(typeof(d.λ)) / d.λ

function entropy(d::Poisson{T}) where T<:Real
    λ = rate(d)
    if λ == zero(T)
        return zero(T)
    elseif λ < 50
        s = zero(T)
        λk = one(T)
        for k = 1:100
            λk *= λ
            s += λk * lgamma(k + 1) / gamma(k + 1)
        end
        return λ * (1 - log(λ)) + exp(-λ) * s
    else
        return log(2 * pi * ℯ * λ)/2 -
               (1 / (12 * λ)) -
               (1 / (24 * λ * λ)) -
               (19 / (360 * λ * λ * λ))
    end
end


### Evaluation

@_delegate_statsfuns Poisson pois λ

struct RecursivePoissonProbEvaluator <: RecursiveProbabilityEvaluator
    λ::Float64
end

RecursivePoissonProbEvaluator(d::Poisson) = RecursivePoissonProbEvaluator(rate(d))
nextpdf(s::RecursivePoissonProbEvaluator, p::Float64, x::Integer) = p * s.λ / x

Base.broadcast!(::typeof(pdf), r::AbstractArray, d::Poisson, rgn::UnitRange) =
    _pdf!(r, d, rgn, RecursivePoissonProbEvaluator(d))
function Base.broadcast(::typeof(pdf), d::Poisson, X::UnitRange)
    r = similar(Array{promote_type(partype(d), eltype(X))}, axes(X))
    r .= pdf.(Ref(d),X)
end


function mgf(d::Poisson, t::Real)
    λ = rate(d)
    return exp(λ * (exp(t) - 1))
end

function cf(d::Poisson, t::Real)
    λ = rate(d)
    return exp(λ * (cis(t) - 1))
end


### Fitting

struct PoissonStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    tw::Float64   # total sample weight
end

suffstats(::Type{Poisson}, x::AbstractArray{T}) where {T<:Integer} = PoissonStats(sum(x), length(x))

function suffstats(::Type{Poisson}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    n = length(x)
    n == length(w) || throw(DimensionMismatch("Inconsistent array lengths."))
    sx = 0.
    tw = 0.
    for i = 1 : n
        @inbounds wi = w[i]
        @inbounds sx += x[i] * wi
        tw += wi
    end
    PoissonStats(sx, tw)
end

fit_mle(::Type{Poisson}, ss::PoissonStats) = Poisson(ss.sx / ss.tw)

## samplers

# TODO: remove RFunctions dependency once Poisson has been fully implemented
# Currently depends on a quantile function for one option
@rand_rdist(Poisson)
rand(d::Poisson) = convert(Int, StatsFuns.RFunctions.poisrand(d.λ))

# algorithm from:
#   J.H. Ahrens, U. Dieter (1982)
#   "Computer Generation of Poisson Deviates from Modified Normal Distributions"
#   ACM Transactions on Mathematical Software, 8(2):163-179
# TODO: implement poisson sampler
function rand(rng::AbstractRNG, d::Poisson)
    μ = d.λ
    if μ >= 10.0  # Case A

        s = sqrt(μ)
        d = 6.0*μ^2
        L = floor(Int64, μ-1.1484)

        # Step N
        T = randn(rng)
        G = μ + s*T

        if G >= 0.0
            K = floor(Int64, G)
            # Step I
            if K >= L
                return K
            end

            # Step S
            U = rand(rng)
            if d*U >= (μ-K)^3
                return K
            end

            # Step P
            px,py,fx,fy = procf(μ,K,s)

            # Step Q
            if fy*(1-U) <= py*exp(px-fx)
                return K
            end
        end

        while true
            # Step E
            E = randexp(rng)
            U = rand(rng)
            U = 2.0*U-1.0
            T = 1.8+copysign(E,U)
            if T <= -0.6744
                continue
            end

            K = floor(Int64, μ + s*T)
            px,py,fx,fy = procf(μ,K,s)
            c = 0.1069/μ

            # Step H
            if c*abs(U) <= py*exp(px+E)-fy*exp(fx+E)
                return K
            end
        end
    else # Case B
        # Ahrens & Dieter use a sequential method for tabulating and looking up quantiles.
        # TODO: check which is more efficient.
        return quantile(d,rand(rng))
    end
end


# Procedure F
function procf(μ,K,s)
    ω = 0.3989422804014327/s
    b1 = 0.041666666666666664/μ
    b2 = 0.3*b1^2
    c3 = 0.14285714285714285*b1*b2
    c2 = b2 - 15.0*c3
    c1 = b1 - 6.0*b2 + 45.0*c3
    c0 = 1.0 - b1 + 3.0*b2 - 15.0*c3

    if K < 10
        px = -μ
        py = μ^K/factorial(K) # replace with loopup?
    else
        δ = 0.08333333333333333/K
        δ -= 4.8*δ^3
        V = (μ-K)/K
        px = K*log1pmx(V) - δ # avoids need for table
        py = 0.3989422804014327/sqrt(K)

    end
    X = (K-μ+0.5)/s
    X2 = X^2
    fx = -0.5*X2 # missing negation in paper
    fy = ω*(((c3*X2+c2)*X2+c1)*X2+c0)
    return px,py,fx,fy
end

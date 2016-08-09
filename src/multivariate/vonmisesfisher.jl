# von Mises-Fisher distribution is useful for directional statistics
#
# The implementation here follows:
#
#   - Wikipedia:
#     http://en.wikipedia.org/wiki/Von_Mises–Fisher_distribution
#
#   - R's movMF package's document:
#     http://cran.r-project.org/web/packages/movMF/vignettes/movMF.pdf
#
#   - Wenzel Jakob's notes:
#     http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
#

immutable VonMisesFisher{T<:Real} <: ContinuousMultivariateDistribution
    μ::Vector{T}
    κ::T
    logCκ::T

    function VonMisesFisher(μ::Vector{T}, κ::T; checknorm::Bool=true)
        if checknorm
            isunitvec(μ) || error("μ must be a unit vector")
        end
        κ > 0 || error("κ must be positive.")
        logCκ = vmflck(length(μ), κ)
        S = promote_type(T, typeof(logCκ))
        new(Vector{S}(μ), S(κ), S(logCκ))
    end
end

VonMisesFisher{T<:Real}(μ::Vector{T}, κ::T) = VonMisesFisher{T}(μ, κ)
VonMisesFisher{T<:Real}(μ::Vector{T}, κ::Real) = VonMisesFisher(promote_eltype(μ, κ)...)

function VonMisesFisher(θ::Vector)
    κ = vecnorm(θ)
    return VonMisesFisher(θ * (1 / κ), κ)
end

show(io::IO, d::VonMisesFisher) = show(io, d, (:μ, :κ))

### Conversions
convert{T<:Real}(::Type{VonMisesFisher{T}}, d::VonMisesFisher) = VonMisesFisher{T}(Vector{T}(d.μ), T(d.κ))
convert{T<:Real}(::Type{VonMisesFisher{T}}, μ::Vector, κ, logCκ) =  VonMisesFisher{T}(Vector{T}(μ), T(κ))



### Basic properties

length(d::VonMisesFisher) = length(d.μ)

meandir(d::VonMisesFisher) = d.μ
concentration(d::VonMisesFisher) = d.κ

insupport{T<:Real}(d::VonMisesFisher, x::AbstractVector{T}) = isunitvec(x)
params(d::VonMisesFisher) = (d.μ, d.κ)
@inline partype{T<:Real}(d::VonMisesFisher{T}) = T

### Evaluation

function _vmflck(p, κ)
    T = typeof(κ)
    hp = T(p/2)
    q = hp - 1
    q * log(κ) - hp * log2π - log(besseli(q, κ))
end
_vmflck3(κ) = log(κ) - log2π - κ - log1mexp(-2κ)
vmflck(p, κ) = (p == 3 ? _vmflck3(κ) : _vmflck(p, κ))

_logpdf{T<:Real}(d::VonMisesFisher, x::AbstractVector{T}) = d.logCκ + d.κ * dot(d.μ, x)


### Sampling

sampler(d::VonMisesFisher) = VonMisesFisherSampler(d.μ, d.κ)

_rand!(d::VonMisesFisher, x::AbstractVector) = _rand!(sampler(d), x)
_rand!(d::VonMisesFisher, x::AbstractMatrix) = _rand!(sampler(d), x)


### Estimation

function fit_mle(::Type{VonMisesFisher}, X::Matrix{Float64})
    r = vec(sum(X, 2))
    n = size(X, 2)
    r_nrm = vecnorm(r)
    μ = scale!(r, 1.0 / r_nrm)
    ρ = r_nrm / n
    κ = _vmf_estkappa(length(μ), ρ)
    VonMisesFisher(μ, κ)
end

fit_mle{T<:Real}(::Type{VonMisesFisher}, X::Matrix{T}) = fit_mle(VonMisesFisher, Float64(X))

function _vmf_estkappa(p::Int, ρ::Float64)
    # Using the fixed-point iteration algorithm in the following paper:
    #
    #   Akihiro Tanabe, Kenji Fukumizu, and Shigeyuki Oba, Takashi Takenouchi, and Shin Ishii
    #   Parameter estimation for von Mises-Fisher distributions.
    #   Computational Statistics, 2007, Vol. 22:145-157.
    #

    const maxiter = 200
    half_p = 0.5 * p

    ρ2 = abs2(ρ)
    κ = ρ * (p - ρ2) / (1 - ρ2)
    i = 0
    while i < maxiter
        i += 1
        κ_prev = κ
        a = (ρ / _vmfA(half_p, κ))
        # println("i = $i, a = $a, abs(a - 1) = $(abs(a - 1))")
        κ *= a
        if abs(a - 1.0) < 1.0e-12
            break
        end
    end
    return κ
end

_vmfA(half_p::Float64, κ::Float64) = besseli(half_p, κ) / besseli(half_p - 1.0, κ)

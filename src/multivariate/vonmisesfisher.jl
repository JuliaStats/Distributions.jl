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

struct VonMisesFisher{T<:Real} <: ContinuousMultivariateDistribution
    μ::Vector{T}
    κ::T
    logCκ::T

    function VonMisesFisher{T}(μ::Vector{T}, κ::T; checknorm::Bool=true) where T
        if checknorm
            isunitvec(μ) || error("μ must be a unit vector")
        end
        κ > 0 || error("κ must be positive.")
        logCκ = vmflck(length(μ), κ)
        S = promote_type(T, typeof(logCκ))
        new{T}(Vector{S}(μ), S(κ), S(logCκ))
    end
end

VonMisesFisher(μ::Vector{T}, κ::T) where {T<:Real} = VonMisesFisher{T}(μ, κ)
function VonMisesFisher(μ::Vector{T}, κ::Real) where {T<:Real}
    R = promote_type(T, eltype(κ))
    return VonMisesFisher(convert(AbstractArray{R}, μ), convert(R, κ))
end

function VonMisesFisher(θ::Vector)
    κ = norm(θ)
    return VonMisesFisher(θ * (1 / κ), κ)
end

show(io::IO, d::VonMisesFisher) = show(io, d, (:μ, :κ))

### Conversions
convert(::Type{VonMisesFisher{T}}, d::VonMisesFisher) where {T<:Real} = VonMisesFisher{T}(convert(Vector{T}, d.μ), T(d.κ); checknorm=false)
Base.convert(::Type{VonMisesFisher{T}}, d::VonMisesFisher{T}) where {T<:Real} = d
convert(::Type{VonMisesFisher{T}}, μ::Vector, κ, logCκ) where {T<:Real} =  VonMisesFisher{T}(convert(Vector{T}, μ), T(κ))



### Basic properties

length(d::VonMisesFisher) = length(d.μ)

meandir(d::VonMisesFisher) = d.μ
concentration(d::VonMisesFisher) = d.κ

insupport(d::VonMisesFisher, x::AbstractVector{T}) where {T<:Real} = isunitvec(x)
params(d::VonMisesFisher) = (d.μ, d.κ)
@inline partype(d::VonMisesFisher{T}) where {T<:Real} = T

### Evaluation

function _vmflck(p, κ)
    T = typeof(κ)
    hp = T(p/2)
    q = hp - 1
    q * log(κ) - hp * log2π - log(besselix(q, κ)) - κ
end
_vmflck3(κ) = log(κ) - log2π - κ - log1mexp(-2κ)
vmflck(p, κ) = (p == 3 ? _vmflck3(κ) : _vmflck(p, κ))

_logpdf(d::VonMisesFisher, x::AbstractVector{T}) where {T<:Real} = d.logCκ + d.κ * dot(d.μ, x)


### Sampling

sampler(d::VonMisesFisher) = VonMisesFisherSampler(d.μ, d.κ)

_rand!(rng::AbstractRNG, d::VonMisesFisher, x::AbstractVector) =
    _rand!(rng, sampler(d), x)
_rand!(rng::AbstractRNG, d::VonMisesFisher, x::AbstractMatrix) =
    _rand!(rng, sampler(d), x)


### Estimation

function fit_mle(::Type{<:VonMisesFisher}, X::Matrix{Float64})
    r = vec(sum(X, dims=2))
    n = size(X, 2)
    r_nrm = norm(r)
    μ = rmul!(r, 1.0 / r_nrm)
    ρ = r_nrm / n
    κ = _vmf_estkappa(length(μ), ρ)
    VonMisesFisher(μ, κ)
end

fit_mle(::Type{<:VonMisesFisher}, X::Matrix{T}) where {T<:Real} = fit_mle(VonMisesFisher, Float64(X))

function _vmf_estkappa(p::Int, ρ::Float64)
    # Using the fixed-point iteration algorithm in the following paper:
    #
    #   Akihiro Tanabe, Kenji Fukumizu, and Shigeyuki Oba, Takashi Takenouchi, and Shin Ishii
    #   Parameter estimation for von Mises-Fisher distributions.
    #   Computational Statistics, 2007, Vol. 22:145-157.
    #

    maxiter = 200
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

_vmfA(half_p::Float64, κ::Float64) = besselix(half_p, κ) / besselix(half_p - 1.0, κ)

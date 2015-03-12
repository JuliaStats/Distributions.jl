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

immutable VonMisesFisher <: ContinuousMultivariateDistribution
    μ::Vector{Float64}
    κ::Float64
    logCκ::Float64

    function VonMisesFisher(μ::Vector{Float64}, κ::Float64; checknorm::Bool=true)
        if checknorm
            isunitvec(μ) || error("μ must be a unit vector")
        end
        κ > 0 || error("κ must be positive.")
        new(μ, κ, vmflck(length(μ), κ))
    end
end

@compat VonMisesFisher{T<:Real}(μ::Vector{T}, κ::Real) = VonMisesFisher(Float64(μ), Float64(κ))

VonMisesFisher(θ::Vector{Float64}) = (κ = vecnorm(θ); VonMisesFisher(scale(θ, 1.0 / κ), κ))
VonMisesFisher{T<:Real}(θ::Vector{T}) = VonMisesFisher(Float64(θ))

show(io::IO, d::VonMisesFisher) = show(io, d, (:μ, :κ))


### Basic properties

length(d::VonMisesFisher) = length(d.μ)

meandir(d::VonMisesFisher) = d.μ
concentration(d::VonMisesFisher) = d.κ

insupport{T<:Real}(d::VonMisesFisher, x::DenseVector{T}) = isunitvec(x)


### Evaluation

function _vmflck(p, κ)
    hp = 0.5 * p
    q = hp - 1.0
    q * log(κ) - hp * log(2π) - log(besseli(q, κ))
end
_vmflck3(κ) = log(κ) - log2π - κ - log1mexp(-2.0 * κ) 
vmflck(p, κ) = (p == 3 ? _vmflck3(κ) : _vmflck(p, κ))::Float64

_logpdf{T<:Real}(d::VonMisesFisher, x::DenseVector{T}) = d.logCκ + d.κ * dot(d.μ, x)


### Sampling 

sampler(d::VonMisesFisher) = VonMisesFisherSampler(d.μ, d.κ)

_rand!(d::VonMisesFisher, x::DenseVector) = _rand!(sampler(d), x)
_rand!(d::VonMisesFisher, x::DenseMatrix) = _rand!(sampler(d), x)


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

@compat fit_mle{T<:Real}(::Type{VonMisesFisher}, X::Matrix{T}) = fit_mle(VonMisesFisher, Float64(X))

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



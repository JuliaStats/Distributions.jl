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

VonMisesFisher{T<:Real}(μ::Vector{T}, κ::Real) = VonMisesFisher(float64(μ), float64(κ))

VonMisesFisher(θ::Vector{Float64}) = (κ = vecnorm(θ); VonMisesFisher(scale(θ, 1.0 / κ), κ))
VonMisesFisher{T<:Real}(θ::Vector{T}) = VonMisesFisher(float64(θ))

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



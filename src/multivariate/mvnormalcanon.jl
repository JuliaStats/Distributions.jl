# Canonical form of multivariate normal

### Generic types

immutable MvNormalCanon{P<:AbstractPDMat,V<:Union{Vector,ZeroVector}} <: AbstractMvNormal
    μ::V    # the mean vector
    h::V    # potential vector, i.e. inv(Σ) * μ
    J::P    # precision matrix, i.e. inv(Σ)
end

typealias FullNormalCanon MvNormalCanon{PDMat{Float64,Matrix{Float64}},Vector{Float64}}
typealias DiagNormalCanon MvNormalCanon{PDiagMat{Float64,Vector{Float64}},Vector{Float64}}
typealias IsoNormalCanon  MvNormalCanon{ScalMat{Float64},Vector{Float64}}

typealias ZeroMeanFullNormalCanon MvNormalCanon{PDMat{Float64,Matrix{Float64}},ZeroVector{Float64}}
typealias ZeroMeanDiagNormalCanon MvNormalCanon{PDiagMat{Float64,Vector{Float64}},ZeroVector{Float64}}
typealias ZeroMeanIsoNormalCanon  MvNormalCanon{ScalMat{Float64},ZeroVector{Float64}}


### Constructors

function MvNormalCanon{P<:AbstractPDMat, T<:Real}(μ::Vector{T}, h::Vector{T}, J::P)
    length(μ) == length(h) == dim(J) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    MvNormalCanon{P,Vector{T}}(μ, h, J)
end

function MvNormalCanon{P<:AbstractPDMat}(J::P)
    z = ZeroVector(Float64, dim(J))
    MvNormalCanon{P,ZeroVector{Float64}}(z, z, J)
end

function MvNormalCanon{P<:AbstractPDMat, T<:Real}(h::Vector{T}, J::P)
    length(h) == dim(J) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    MvNormalCanon{P,Vector{T}}(J \ h, h, J)
end

MvNormalCanon(h::Vector, J::Matrix) = MvNormalCanon(h, PDMat(J))
MvNormalCanon(h::Vector, prec::Vector) = MvNormalCanon(h, PDiagMat(prec))
MvNormalCanon(h::Vector, prec) = MvNormalCanon(h, ScalMat(length(h), prec))

MvNormalCanon(J::Matrix) = MvNormalCanon(PDMat(J))
MvNormalCanon(prec::Vector) = MvNormalCanon(PDiagMat(prec))
MvNormalCanon(d::Int, prec) = MvNormalCanon(ScalMat(d, prec))


### Show

distrname(d::IsoNormalCanon) = "IsoNormalCanon"
distrname(d::DiagNormalCanon) = "DiagNormalCanon"
distrname(d::FullNormalCanon) = "FullNormalCanon"

distrname(d::ZeroMeanIsoNormalCanon) = "ZeroMeanIsoNormalCanon"
distrname(d::ZeroMeanDiagNormalCanon) = "ZeroMeanDiagormalCanon"
distrname(d::ZeroMeanFullNormalCanon) = "ZeroMeanFullNormalCanon"

### conversion between conventional form and canonical form

meanform{C,V}(d::MvNormalCanon{C,V}) = MvNormal{C,V}(d.μ, inv(d.J))

canonform{C, T<:Real}(d::MvNormal{C,Vector{T}}) = (J = inv(d.Σ); MvNormalCanon(d.μ, J * d.μ, J))
canonform{C, T<:Real}(d::MvNormal{C,ZeroVector{T}}) = MvNormalCanon(inv(d.Σ))


### Basic statistics

length(d::MvNormalCanon) = length(d.μ)
mean(d::MvNormalCanon) = convert(Vector{eltype(d.μ)}, d.μ)
params(d::MvNormalCanon) = (d.μ, d.h, d.J)

var(d::MvNormalCanon) = diag(inv(d.J))
cov(d::MvNormalCanon) = full(inv(d.J))
invcov(d::MvNormalCanon) = full(d.J)
logdetcov(d::MvNormalCanon) = -logdet(d.J)


### Evaluation

sqmahal(d::MvNormalCanon, x::AbstractVector) = quad(d.J, x - d.μ)
sqmahal!(r::AbstractVector, d::MvNormalCanon, x::AbstractMatrix) = quad!(r, d.J, x .- d.μ)


# Sampling (for GenericMvNormal)

unwhiten_winv!(J::AbstractPDMat, x::AbstractVecOrMat) = unwhiten!(inv(J), x)
unwhiten_winv!(J::PDiagMat, x::AbstractVecOrMat) = whiten!(J, x)
unwhiten_winv!(J::ScalMat, x::AbstractVecOrMat) = whiten!(J, x)

_rand!(d::MvNormalCanon, x::AbstractVecOrMat) = add!(unwhiten_winv!(d.J, randn!(x)), d.μ)

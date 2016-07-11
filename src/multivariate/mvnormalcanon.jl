# Canonical form of multivariate normal

### Generic types

immutable MvNormalCanon{T<:Real,P<:AbstractPDMat,V<:Union{Vector,ZeroVector}} <: AbstractMvNormal
    μ::V    # the mean vector
    h::V    # potential vector, i.e. inv(Σ) * μ
    J::P    # precision matrix, i.e. inv(Σ)
end

typealias FullNormalCanon MvNormalCanon{Float64, PDMat{Float64,Matrix{Float64}},Vector{Float64}}
typealias DiagNormalCanon MvNormalCanon{Float64,PDiagMat{Float64,Vector{Float64}},Vector{Float64}}
typealias IsoNormalCanon  MvNormalCanon{Float64,ScalMat{Float64},Vector{Float64}}

typealias ZeroMeanFullNormalCanon MvNormalCanon{Float64,PDMat{Float64,Matrix{Float64}},ZeroVector{Float64}}
typealias ZeroMeanDiagNormalCanon MvNormalCanon{Float64,PDiagMat{Float64,Vector{Float64}},ZeroVector{Float64}}
typealias ZeroMeanIsoNormalCanon  MvNormalCanon{Float64,ScalMat{Float64},ZeroVector{Float64}}


### Constructors

function MvNormalCanon{T<:Real}(μ::Vector{T}, h::Vector{T}, J::AbstractPDMat{T})
    length(μ) == length(h) == dim(J) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    MvNormalCanon{T,typeof(J),typeof(μ)}(μ, h, J)
end

function MvNormalCanon{T<:Real, P<:AbstractPDMat}(μ::Vector{T}, h::Vector{T}, J::P)
    R = promote_type(T, eltype(J))
    MvNormalCanon(Vector{R}(μ), promote_eltype(h, J)...)
end

MvNormalCanon{T<:Real, S<:Real, P<:AbstractPDMat}(μ::Vector{T}, h::Vector{S}, J::P) = MvNormalCanon(promote_eltype(μ, h)..., J)

function MvNormalCanon{P<:AbstractPDMat}(J::P)
    z = ZeroVector(eltype(J), dim(J))
    MvNormalCanon{eltype(J),P,ZeroVector{eltype(J)}}(z, z, J)
end

function MvNormalCanon{T<:Real, P<:AbstractPDMat}(h::Vector{T}, J::P)
    length(h) == dim(J) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    hh, JJ = promote_eltype(h, J)
    MvNormalCanon{eltype(hh),typeof(JJ),typeof(hh)}(JJ \ hh, hh, JJ)
end

MvNormalCanon{T<:Real}(h::Vector{T}, J::Matrix{T}) = MvNormalCanon(h, PDMat(J))
MvNormalCanon{T<:Real}(h::Vector{T}, prec::Vector{T}) = MvNormalCanon(h, PDiagMat(prec))
MvNormalCanon{T<:Real}(h::Vector{T}, prec::T) = MvNormalCanon(h, ScalMat(length(h), prec))

MvNormalCanon{T<:Real, S<:Real}(h::Vector{T}, J::VecOrMat{S}) = MvNormalCanon(promote_eltype(h, J)...)
MvNormalCanon{T<:Real, S<:Real}(h::Vector{T}, prec::S) = MvNormalCanon(promote_eltype(h, prec)...)

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

meanform(d::MvNormalCanon) = MvNormal(d.μ, inv(d.J))
# meanform{C, T<:Real}(d::MvNormalCanon{T,C,Vector{T}}) = MvNormal(d.μ, inv(d.J))
# meanform{C, T<:Real}(d::MvNormalCanon{T,C,ZeroVector{T}}) = MvNormal(inv(d.J))

canonform{C, T<:Real}(d::MvNormal{T,C,Vector{T}}) = (J = inv(d.Σ); MvNormalCanon(d.μ, J * d.μ, J))
canonform{C, T<:Real}(d::MvNormal{T,C,ZeroVector{T}}) = MvNormalCanon(inv(d.Σ))

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

_rand!(d::MvNormalCanon, x::AbstractMatrix) = add!(unwhiten_winv!(d.J, randn!(x)), d.μ)

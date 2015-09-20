# Canonical form of multivariate normal

### Generic types

@compat immutable MvNormalCanon{P<:AbstractPDMat,V<:Union{Vector{Float64},ZeroVector{Float64}}} <: AbstractMvNormal
    μ::V    # the mean vector
    h::V    # potential vector, i.e. inv(Σ) * μ
    J::P    # precision matrix, i.e. inv(Σ)
end

typealias FullNormalCanon MvNormalCanon{PDMat,Vector{Float64}}
typealias DiagNormalCanon MvNormalCanon{PDiagMat,Vector{Float64}}
typealias IsoNormalCanon  MvNormalCanon{ScalMat,Vector{Float64}}

typealias ZeroMeanFullNormalCanon MvNormalCanon{PDMat,ZeroVector{Float64}}
typealias ZeroMeanDiagNormalCanon MvNormalCanon{PDiagMat,ZeroVector{Float64}}
typealias ZeroMeanIsoNormalCanon  MvNormalCanon{ScalMat,ZeroVector{Float64}}


### Constructors

function MvNormalCanon{P<:AbstractPDMat}(μ::Vector{Float64}, h::Vector{Float64}, J::P)
    length(μ) == length(h) == dim(J) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    MvNormalCanon{P,Vector{Float64}}(μ, h, J)
end

function MvNormalCanon{P<:AbstractPDMat}(J::P)
    z = ZeroVector(Float64, dim(J))
    MvNormalCanon{P,ZeroVector{Float64}}(z, z, J)
end

function MvNormalCanon{P<:AbstractPDMat}(h::Vector{Float64}, J::P)
    length(h) == dim(J) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    MvNormalCanon{P,Vector{Float64}}(J \ h, h, J)
end

MvNormalCanon(h::Vector{Float64}, J::Matrix{Float64}) = MvNormalCanon(h, PDMat(J))
MvNormalCanon(h::Vector{Float64}, prec::Vector{Float64}) = MvNormalCanon(h, PDiagMat(prec))
MvNormalCanon(h::Vector{Float64}, prec::Float64) = MvNormalCanon(h, ScalMat(length(h), prec))

MvNormalCanon(J::Matrix{Float64}) = MvNormalCanon(PDMat(J))
MvNormalCanon(prec::Vector{Float64}) = MvNormalCanon(PDiagMat(prec))
MvNormalCanon(d::Int, prec::Float64) = MvNormalCanon(ScalMat(d, prec))


### Show

distrname(d::IsoNormalCanon) = "IsoNormalCanon"
distrname(d::DiagNormalCanon) = "DiagNormalCanon"
distrname(d::FullNormalCanon) = "FullNormalCanon"

distrname(d::ZeroMeanIsoNormalCanon) = "ZeroMeanIsoNormalCanon"
distrname(d::ZeroMeanDiagNormalCanon) = "ZeroMeanDiagormalCanon"
distrname(d::ZeroMeanFullNormalCanon) = "ZeroMeanFullNormalCanon"

### conversion between conventional form and canonical form

meanform{C,V}(d::MvNormalCanon{C,V}) = MvNormal{C,V}(d.μ, inv(d.J))

canonform{C}(d::MvNormal{C,Vector{Float64}}) = (J = inv(d.Σ); MvNormalCanon(d.μ, J * d.μ, J))
canonform{C}(d::MvNormal{C,ZeroVector{Float64}}) = MvNormalCanon(inv(d.Σ))


### Basic statistics

length(d::MvNormalCanon) = length(d.μ)
mean(d::MvNormalCanon) = convert(Vector{Float64}, d.μ)

var(d::MvNormalCanon) = diag(inv(d.J))
cov(d::MvNormalCanon) = full(inv(d.J))
invcov(d::MvNormalCanon) = full(d.J)
logdetcov(d::MvNormalCanon) = -logdet(d.J)


### Evaluation

sqmahal(d::MvNormalCanon, x::DenseVector{Float64}) = quad(d.J, x - d.μ)
sqmahal!(r::DenseVector{Float64}, d::MvNormalCanon, x::DenseMatrix{Float64}) = quad!(r, d.J, x .- d.μ)


# Sampling (for GenericMvNormal)

unwhiten_winv!(J::AbstractPDMat, x::DenseVecOrMat) = unwhiten!(inv(J), x)
unwhiten_winv!(J::PDiagMat, x::DenseVecOrMat) = whiten!(J, x)
unwhiten_winv!(J::ScalMat, x::DenseVecOrMat) = whiten!(J, x)

_rand!(d::MvNormalCanon, x::DenseVecOrMat) = add!(unwhiten_winv!(d.J, randn!(x)), d.μ)

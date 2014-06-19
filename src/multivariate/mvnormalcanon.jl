# Canonical form of multivariate normal

## generic types

immutable GenericMvNormalCanon{Prec<:AbstractPDMat} <: AbstractMvNormal
	dim::Int             # dimension of sample space
	zeromean::Bool       # whether the mean vector is zero
	μ::Vector{Float64}   # the mean vector
    h::Vector{Float64}   # potential vector, i.e. inv(Σ) * μ
    J::Prec              # precision matrix, i.e. inv(Σ)

    function GenericMvNormalCanon(μ::Vector{Float64}, h::Vector{Float64}, J::Prec, zmean::Bool)
    	d = dim(J)
    	length(μ) == length(h) == d || throw(ArgumentError("Inconsistent argument dimensions."))
    	new(d, zmean, μ, h, J)
    end  

	function GenericMvNormalCanon(J::Prec)
		d = dim(J)
    	new(d, true, zeros(d), zeros(d), J)
    end 
end

function GenericMvNormalCanon{P<:AbstractPDMat}(μ::Vector{Float64}, h::Vector{Float64}, J::P)
	GenericMvNormalCanon{P}(μ, h, J, allzeros(μ))
end

function GenericMvNormalCanon{P<:AbstractPDMat}(h::Vector{Float64}, J::P, zmean::Bool)
	μ = zmean ? zeros(length(h)) : (J \ h)
	GenericMvNormalCanon{P}(μ, h, J, zmean)
end

function GenericMvNormalCanon{P<:AbstractPDMat}(h::Vector{Float64}, J::P)
	GenericMvNormalCanon(h, J, allzeros(h))
end

function GenericMvNormalCanon{P<:AbstractPDMat}(J::P)
    d = dim(J)
    GenericMvNormalCanon{P}(zeros(d), zeros(d), J, true)
end

## type aliases and convenient constructors

typealias MvNormalCanon   GenericMvNormalCanon{PDMat} 
typealias DiagNormalCanon GenericMvNormalCanon{PDiagMat} 
typealias IsoNormalCanon  GenericMvNormalCanon{ScalMat}

MvNormalCanon(J::PDMat) = GenericMvNormalCanon(J)
MvNormalCanon(J::Matrix{Float64}) = GenericMvNormalCanon(PDMat(J))
MvNormalCanon(h::Vector{Float64}, J::PDMat) = GenericMvNormalCanon(h, J)
MvNormalCanon(h::Vector{Float64}, J::Matrix{Float64}) = GenericMvNormalCanon(h, PDMat(J))

DiagNormalCanon(J::PDiagMat) = GenericMvNormalCanon(J)
DiagNormalCanon(J::Vector{Float64}) = GenericMvNormalCanon(PDiagMat(J))
DiagNormalCanon(h::Vector{Float64}, J::PDiagMat) = GenericMvNormalCanon(h, J)
DiagNormalCanon(h::Vector{Float64}, J::Vector{Float64}) = GenericMvNormalCanon(h, PDiagMat(J))

IsoNormalCanon(J::ScalMat) = GenericMvNormalCanon(J)
IsoNormalCanon(d::Integer, prec::Real) = GenericMvNormalCanon(ScalMat(int(d), float64(prec)))
IsoNormalCanon(h::Vector{Float64}, J::ScalMat) = GenericMvNormalCanon(h, J)
IsoNormalCanon(h::Vector{Float64}, prec::Real) = GenericMvNormalCanon(h, ScalMat(length(h), float64(prec)))

# conversion between conventional form and canonical form

function Base.convert{C<:AbstractPDMat}(D::Type{GenericMvNormal{C}}, cf::GenericMvNormalCanon{C})
	GenericMvNormal{C}(cf.dim, cf.zeromean, cf.μ, inv(cf.J))
end

function Base.convert{C<:AbstractPDMat}(D::Type{GenericMvNormalCanon{C}}, d::GenericMvNormal{C})
	J::C = inv(d.Σ)
	h::Vector{Float64} = J * d.μ
	GenericMvNormalCanon{C}(d.μ, h, J, d.zeromean)
end

canonform{C<:AbstractPDMat}(d::GenericMvNormal{C}) = convert(GenericMvNormalCanon{C}, d)


# Basic statistics

length(d::GenericMvNormalCanon) = d.dim

mean(d::GenericMvNormalCanon) = d.μ
mode(d::GenericMvNormalCanon) = d.μ
modes(d::GenericMvNormalCanon) = [mode(d)]

var(d::GenericMvNormalCanon) = diag(inv(d.J))
cov(d::GenericMvNormalCanon) = full(inv(d.J))
invcov(d::GenericMvNormalCanon) = full(d.J)
logdet_cov(d::GenericMvNormalCanon) = -logdet(d.J)

entropy(d::GenericMvNormalCanon) = 0.5 * (length(d) * (float64(log2π) + 1.0) - logdet(d.J))


# PDF evaluation

function sqmahal(d::GenericMvNormalCanon, x::Vector{Float64}) 
    z::Vector{Float64} = d.zeromean ? x : x - d.μ
    quad(d.J, z) 
end

function sqmahal!(r::Array{Float64}, d::GenericMvNormalCanon, x::Matrix{Float64})
    if !(size(x, 1) == length(d) && size(x, 2) == length(r))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    z::Matrix{Float64} = d.zeromean ? x : bsubtract(x, d.μ, 1)
    quad!(r, d.J, z)
end


# Sampling (for GenericMvNormal)

function _rand!(d::GenericMvNormalCanon, x::DenseVector{Float64})
    unwhiten_winv!(d.J, randn!(x))
    if !d.zeromean
        add!(x, d.μ)
    end
    x
end

function _rand!(d::GenericMvNormalCanon, x::DenseMatrix{Float64})
    unwhiten_winv!(d.J, randn!(x))
    if !d.zeromean
        badd!(x, d.μ, 1)
    end
    x
end


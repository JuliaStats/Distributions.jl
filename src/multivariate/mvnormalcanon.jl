# Canonical form of multivariate normal

### Generic types

"""
    MvNormalCanon

The multivariate normal distribution is an [exponential family distribution](http://en.wikipedia.org/wiki/Exponential_family),
with two *canonical parameters*: the *potential vector* ``\\mathbf{h}`` and the *precision matrix* ``\\mathbf{J}``.
The relation between these parameters and the conventional representation (*i.e.* the one using mean ``\\boldsymbol{\\mu}`` and
covariance ``\\boldsymbol{\\Sigma}``) is:

```math
\\mathbf{h} = \\boldsymbol{\\Sigma}^{-1} \\boldsymbol{\\mu}, \\quad \\text{ and } \\quad \\mathbf{J} = \\boldsymbol{\\Sigma}^{-1}
```

The canonical parameterization is widely used in Bayesian analysis. We provide a type `MvNormalCanon`,
which is also a subtype of `AbstractMvNormal` to represent a multivariate normal distribution using
canonical parameters. Particularly, `MvNormalCanon` is defined as:

```julia
struct MvNormalCanon{T<:Real,P<:AbstractPDMat,V<:AbstractVector} <: AbstractMvNormal
    μ::V    # the mean vector
    h::V    # potential vector, i.e. inv(Σ) * μ
    J::P    # precision matrix, i.e. inv(Σ)
end
```

We also define aliases for common specializations of this parametric type:

```julia
const FullNormalCanon = MvNormalCanon{Float64, PDMat{Float64,Matrix{Float64}},    Vector{Float64}}
const DiagNormalCanon = MvNormalCanon{Float64, PDiagMat{Float64,Vector{Float64}}, Vector{Float64}}
const IsoNormalCanon  = MvNormalCanon{Float64, ScalMat{Float64},                  Vector{Float64}}

const ZeroMeanFullNormalCanon{Axes} = MvNormalCanon{Float64, PDMat{Float64,Matrix{Float64}},    Zeros{Float64,1,Axes}}
const ZeroMeanDiagNormalCanon{Axes} = MvNormalCanon{Float64, PDiagMat{Float64,Vector{Float64}}, Zeros{Float64,1,Axes}}
const ZeroMeanIsoNormalCanon{Axes}  = MvNormalCanon{Float64, ScalMat{Float64},                  Zeros{Float64,1,Axes}}
```

**Note:** `MvNormalCanon` share the same set of methods as `MvNormal`.
"""
struct MvNormalCanon{T<:Real,P<:AbstractPDMat,V<:AbstractVector} <: AbstractMvNormal
    μ::V    # the mean vector
    h::V    # potential vector, i.e. inv(Σ) * μ
    J::P    # precision matrix, i.e. inv(Σ)
end

const FullNormalCanon = MvNormalCanon{Float64,PDMat{Float64,Matrix{Float64}},Vector{Float64}}
const DiagNormalCanon = MvNormalCanon{Float64,PDiagMat{Float64,Vector{Float64}},Vector{Float64}}
const IsoNormalCanon  = MvNormalCanon{Float64,ScalMat{Float64},Vector{Float64}}

const ZeroMeanFullNormalCanon{Axes} = MvNormalCanon{Float64,PDMat{Float64,Matrix{Float64}},Zeros{Float64,1,Axes}}
const ZeroMeanDiagNormalCanon{Axes} = MvNormalCanon{Float64,PDiagMat{Float64,Vector{Float64}},Zeros{Float64,1,Axes}}
const ZeroMeanIsoNormalCanon{Axes}  = MvNormalCanon{Float64,ScalMat{Float64},Zeros{Float64,1,Axes}}


### Constructors
function MvNormalCanon(μ::AbstractVector{T}, h::AbstractVector{T}, J::AbstractPDMat{T}) where {T<:Real}
    length(μ) == length(h) == size(J, 1) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    if typeof(μ) === typeof(h)
        return MvNormalCanon{T,typeof(J),typeof(μ)}(μ, h, J)
    else
        return MvNormalCanon{T,typeof(J),Vector{T}}(collect(μ), collect(h), J)
    end
end

function MvNormalCanon(μ::AbstractVector{T}, h::AbstractVector{T}, J::AbstractPDMat) where {T<:Real}
    R = promote_type(T, eltype(J))
    MvNormalCanon(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, h), convert(AbstractArray{R}, J))
end

function MvNormalCanon(μ::AbstractVector{<:Real}, h::AbstractVector{<:Real}, J::AbstractPDMat)
    R = Base.promote_eltype(μ, h, J)
    MvNormalCanon(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, h), convert(AbstractArray{R}, J))
end

function MvNormalCanon(h::AbstractVector{<:Real}, J::AbstractPDMat)
    length(h) == size(J, 1) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    R = Base.promote_eltype(h, J)
    hh = convert(AbstractArray{R}, h)
    JJ = convert(AbstractArray{R}, J)
    MvNormalCanon(JJ \ hh, hh, JJ)
end

"""
    MvNormalCanon(h::AbstractVector{<:Real}, J::AbstractMatrix{<:Real})

Construct a multivariate normal distribution with potential vector `h` and precision matrix
`J`.
"""
MvNormalCanon(h::AbstractVector{<:Real}, J::AbstractMatrix{<:Real}) = MvNormalCanon(h, PDMat(J))
MvNormalCanon(h::AbstractVector{<:Real}, J::Diagonal{<:Real}) = MvNormalCanon(h, PDiagMat(J.diag))
MvNormalCanon(μ::AbstractVector{<:Real}, J::Union{Symmetric{<:Real,<:Diagonal{<:Real}},Hermitian{<:Real,<:Diagonal{<:Real}}}) = MvNormalCanon(μ, PDiagMat(J.data.diag))
function MvNormalCanon(h::AbstractVector{<:Real}, J::UniformScaling{<:Real})
    return MvNormalCanon(h, ScalMat(length(h), J.λ))
end
function MvNormalCanon(
    h::AbstractVector{<:Real}, J::Diagonal{<:Real,<:FillArrays.AbstractFill{<:Real,1}}
)
    return MvNormalCanon(h, ScalMat(size(J, 1), FillArrays.getindex_value(J.diag)))
end

# Constructor without mean vector
"""
    MvNormalCanon(J::AbstractMatrix{<:Real})

Construct a multivariate normal distribution with zero mean (thus zero potential vector) and
precision matrix `J`.
"""
MvNormalCanon(J::AbstractMatrix{<:Real}) = MvNormalCanon(Zeros{eltype(J)}(size(J, 1)), J)

# Deprecated constructors
Base.@deprecate MvNormalCanon(h::AbstractVector{<:Real}, prec::AbstractVector{<:Real}) MvNormalCanon(h, LinearAlgebra.Diagonal(prec))
Base.@deprecate MvNormalCanon(h::AbstractVector{<:Real}, prec::Real) MvNormalCanon(h, prec * I)
Base.@deprecate MvNormalCanon(prec::AbstractVector) MvNormalCanon(LinearAlgebra.Diagonal(prec))
Base.@deprecate MvNormalCanon(d::Int, prec::Real) MvNormalCanon(LinearAlgebra.Diagonal(FillArrays.Fill(prec, d)))

### Show

distrname(d::IsoNormalCanon) = "IsoNormalCanon"
distrname(d::DiagNormalCanon) = "DiagNormalCanon"
distrname(d::FullNormalCanon) = "FullNormalCanon"

distrname(d::ZeroMeanIsoNormalCanon) = "ZeroMeanIsoNormalCanon"
distrname(d::ZeroMeanDiagNormalCanon) = "ZeroMeanDiagormalCanon"
distrname(d::ZeroMeanFullNormalCanon) = "ZeroMeanFullNormalCanon"

### Conversion
function convert(::Type{MvNormalCanon{T}}, d::MvNormalCanon) where {T<:Real}
    MvNormalCanon(convert(AbstractArray{T}, d.μ), convert(AbstractArray{T}, d.h), convert(AbstractArray{T}, d.J))
end
Base.convert(::Type{MvNormalCanon{T}}, d::MvNormalCanon{T}) where {T<:Real} = d

function convert(::Type{MvNormalCanon{T}}, μ::AbstractVector{<:Real}, h::AbstractVector{<:Real}, J::AbstractPDMat) where {T<:Real}
    MvNormalCanon(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, h), convert(AbstractArray{T}, J))
end

### conversion between conventional form and canonical form

meanform(d::MvNormalCanon) = MvNormal(d.μ, inv(d.J))
# meanform{C, T<:Real}(d::MvNormalCanon{T,C,Vector{T}}) = MvNormal(d.μ, inv(d.J))
# meanform{C, T<:Real}(d::MvNormalCanon{T,C,Zeros{T}}) = MvNormal(inv(d.J))

function canonform(d::MvNormal{T,C,<:AbstractVector{T}}) where {C, T<:Real}
    J = inv(d.Σ)
    return MvNormalCanon(d.μ, J * collect(d.μ), J)
end
canonform(d::MvNormal{T,C,Zeros{T}}) where {C, T<:Real} = MvNormalCanon(inv(d.Σ))

### Basic statistics

length(d::MvNormalCanon) = length(d.μ)
mean(d::MvNormalCanon) = convert(Vector{eltype(d.μ)}, d.μ)
params(d::MvNormalCanon) = (d.μ, d.h, d.J)
@inline partype(d::MvNormalCanon{T}) where {T<:Real} = T
Base.eltype(::Type{<:MvNormalCanon{T}}) where {T} = T

var(d::MvNormalCanon) = diag(inv(d.J))
cov(d::MvNormalCanon) = Matrix(inv(d.J))
invcov(d::MvNormalCanon) = Matrix(d.J)
logdetcov(d::MvNormalCanon) = -logdet(d.J)


### Evaluation

sqmahal(d::MvNormalCanon, x::AbstractVector) = quad(d.J, broadcast(-, x, d.μ))
sqmahal!(r::AbstractVector, d::MvNormalCanon, x::AbstractMatrix) = quad!(r, d.J, broadcast(-, x, d.μ))


# Sampling (for GenericMvNormal)

unwhiten_winv!(J::AbstractPDMat, x::AbstractVecOrMat) = unwhiten!(inv(J), x)
unwhiten_winv!(J::PDiagMat, x::AbstractVecOrMat) = whiten!(J, x)
unwhiten_winv!(J::ScalMat, x::AbstractVecOrMat) = whiten!(J, x)
if isdefined(PDMats, :PDSparseMat)
    unwhiten_winv!(J::PDSparseMat, x::AbstractVecOrMat) = x[:] = J.chol.PtL' \ x
end

function _rand!(rng::AbstractRNG, d::MvNormalCanon, x::AbstractVector)
    unwhiten_winv!(d.J, randn!(rng, x))
    x .+= d.μ
    return x
end
function _rand!(rng::AbstractRNG, d::MvNormalCanon, x::AbstractMatrix)
    unwhiten_winv!(d.J, randn!(rng, x))
    x .+= d.μ
    return x
end

"""
    InverseWishart <: ContinuousMatrixDistribution

The *inverse Wishart* probability distribution.

# Constructors

    InverseWishart(ν|nu|dof=, Ψ|Psi|scale=)

Construct an `InverseWishart` distribution object with `ν` degrees of freedom, and scale parameter `Ψ`.

    InverseWishart(ν|nu|dof=, mean=)
    InverseWishart(ν|nu|dof=, mode=)

Construct an `InverseWishart` distribution object with `ν` degrees of freedom, and matching the relevant `mean` or `mode`.

# Details

The inverse Wishart is the distribution of the inverse of a [`Wishart`](@ref) variate, with degrees of freedom `ν` and scale matrix `inv(Φ)`.

It is a conjugate prior for the covariance matrix of the [`MvNormal`](@ref) distribution.

# External links

- [Inverse Wishart distribution on Wikipedia](http://en.wikipedia.org/wiki/Inverse-Wishart_distribution)
"""
struct InverseWishart{T<:Real, ST<:AbstractPDMat} <: ContinuousMatrixDistribution
    ν::T      # degree of freedom
    Ψ::ST     # scale matrix
    c0::T     # log of normalizing constant
end

#### Constructors

function InverseWishart(ν::T, Ψ::AbstractPDMat{T}) where T<:Real
    p = dim(Ψ)
    @check_args(InverseWishart, ν > p-1)
    c0 = _invwishart_c0(ν, Ψ)
    R = Base.promote_eltype(T, c0)
    prom_Ψ = convert(AbstractArray{R}, Ψ)
    InverseWishart{R, typeof(prom_Ψ)}(R(ν), prom_Ψ, R(c0))
end

function InverseWishart(ν::Real, Ψ::AbstractPDMat)
    T = Base.promote_eltype(ν, Ψ)
    InverseWishart(T(ν), convert(AbstractArray{T}, Ψ))
end

InverseWishart(ν::Real, Ψ::Matrix) = InverseWishart(ν, PDMat(Ψ))

InverseWishart(ν::Real, Ψ::Cholesky) = InverseWishart(ν, PDMat(Ψ))

@kwdispatch (::Type{D})(;nu=>ν,dof=>ν,Psi=>Ψ,scale=>Ψ) where {D<:InverseWishart} begin
    (ν, Ψ) -> D(ν, Ψ)
    function (ν, mean)
        p = dim(mean)
        @check_args(InverseWishart, ν > p+1)
        D(ν, mean ./ (ν-p-1))
    end
    function (ν, mode)
        p = dim(mode)
        D(ν, mode ./ (ν+p+1))
    end
end


function _invwishart_c0(ν::Real, Ψ::AbstractPDMat)
    h_ν = ν / 2
    p = dim(Ψ)
    h_ν * (p * typeof(ν)(logtwo) - logdet(Ψ)) + logmvgamma(p, h_ν)
end


#### Properties

insupport(::Type{InverseWishart}, X::Matrix) = isposdef(X)
insupport(d::InverseWishart, X::Matrix) = size(X) == size(d) && isposdef(X)

dim(d::InverseWishart) = dim(d.Ψ)
size(d::InverseWishart) = (p = dim(d); (p, p))
size(d::InverseWishart, i) = size(d)[i]
params(d::InverseWishart) = (d.ν, d.Ψ, d.c0)
@inline partype(d::InverseWishart{T}) where {T<:Real} = T

### Conversion
function convert(::Type{InverseWishart{T}}, d::InverseWishart) where T<:Real
    P = Wishart{T}(d.Ψ)
    InverseWishart{T, typeof(P)}(T(d.ν), P, T(d.c0))
end
function convert(::Type{InverseWishart{T}}, ν, Ψ::AbstractPDMat, c0) where T<:Real
    P = Wishart{T}(Ψ)
    InverseWishart{T, typeof(P)}(T(ν), P, T(c0))
end

#### Show

show(io::IO, d::InverseWishart) = show_multline(io, d, [(:ν, d.ν), (:Ψ, Matrix(d.Ψ))])


#### Statistics

function mean(d::InverseWishart)
    ν = d.ν
    p = dim(d)
    r = ν - (p + 1)
    if r > 0.0
        return Matrix(d.Ψ) * (1.0 / r)
    else
        error("mean only defined for ν > p + 1")
    end
end

mode(d::InverseWishart) = d.Ψ * inv(d.ν + dim(d) + 1.0)


#### Evaluation

function _logpdf(d::InverseWishart, X::AbstractMatrix)
    p = dim(d)
    ν = d.ν
    Xcf = cholesky(X)
    # we use the fact: tr(Ψ * inv(X)) = tr(inv(X) * Ψ) = tr(X \ Ψ)
    Ψ = Matrix(d.Ψ)
    -0.5 * ((ν + p + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end


#### Sampling

function _rand!(rng::AbstractRNG, d::InverseWishart, A::AbstractMatrix)
    A .= inv(cholesky!(_rand!(rng, Wishart(d.ν, inv(d.Ψ)), A))))
end

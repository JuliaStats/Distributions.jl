"""
    InverseWishart(ν, Ψ)
```julia
ν::Real           degrees of freedom (greater than p - 1)
Ψ::AbstractPDMat  p x p scale matrix
```
The [inverse Wishart distribution](http://en.wikipedia.org/wiki/Inverse-Wishart_distribution)
generalizes the inverse gamma distribution to ``p\\times p`` real, positive definite
matrices ``\\boldsymbol{\\Sigma}``. If ``\\boldsymbol{\\Sigma}\\sim IW_p(\\nu,\\boldsymbol{\\Psi})``,
then its probability density function is

```math
f(\\boldsymbol{\\Sigma}; \\nu,\\boldsymbol{\\Psi}) =
\\frac{\\left|\\boldsymbol{\\Psi}\\right|^{\\nu/2}}{2^{\\nu p/2}\\Gamma_p(\\frac{\\nu}{2})} \\left|\\boldsymbol{\\Sigma}\\right|^{-(\\nu+p+1)/2} e^{-\\frac{1}{2}\\operatorname{tr}(\\boldsymbol{\\Psi}\\boldsymbol{\\Sigma}^{-1})}.
```

``\\mathbf{H}\\sim W_p(\\nu, \\mathbf{S})`` if and only if ``\\mathbf{H}^{-1}\\sim IW_p(\\nu, \\mathbf{S}^{-1})``.
"""
struct InverseWishart{T<:Real, ST<:AbstractPDMat} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    Ψ::ST     # scale matrix
    c0::T     # log of normalizing constant
end

#### Constructors

function InverseWishart(df::T, Ψ::AbstractPDMat{T}) where T<:Real
    p = dim(Ψ)
    df > p - 1 || error("df should be greater than dim - 1.")
    c0 = _invwishart_c0(df, Ψ)
    R = Base.promote_eltype(T, c0)
    prom_Ψ = convert(AbstractArray{R}, Ψ)
    InverseWishart{R, typeof(prom_Ψ)}(R(df), prom_Ψ, R(c0))
end

function InverseWishart(df::Real, Ψ::AbstractPDMat)
    T = Base.promote_eltype(df, Ψ)
    InverseWishart(T(df), convert(AbstractArray{T}, Ψ))
end

InverseWishart(df::Real, Ψ::Matrix) = InverseWishart(df, PDMat(Ψ))

InverseWishart(df::Real, Ψ::Cholesky) = InverseWishart(df, PDMat(Ψ))

function _invwishart_c0(df::Real, Ψ::AbstractPDMat)
    h_df = df / 2
    p = dim(Ψ)
    h_df * (p * typeof(df)(logtwo) - logdet(Ψ)) + logmvgamma(p, h_df)
end

#### Properties

insupport(::Type{InverseWishart}, X::Matrix) = isposdef(X)
insupport(d::InverseWishart, X::Matrix) = size(X) == size(d) && isposdef(X)

dim(d::InverseWishart) = dim(d.Ψ)
size(d::InverseWishart) = (p = dim(d); (p, p))
size(d::InverseWishart, i) = size(d)[i]
rank(d::InverseWishart) = dim(d)
params(d::InverseWishart) = (d.df, d.Ψ, d.c0)
@inline partype(d::InverseWishart{T}) where {T<:Real} = T

### Conversion
function convert(::Type{InverseWishart{T}}, d::InverseWishart) where T<:Real
    P = convert(AbstractArray{T}, d.Ψ)
    InverseWishart{T, typeof(P)}(T(d.df), P, T(d.c0))
end
function convert(::Type{InverseWishart{T}}, df, Ψ::AbstractPDMat, c0) where T<:Real
    P = convert(AbstractArray{T}, Ψ)
    InverseWishart{T, typeof(P)}(T(df), P, T(c0))
end

#### Show

show(io::IO, d::InverseWishart) = show_multline(io, d, [(:df, d.df), (:Ψ, Matrix(d.Ψ))])


#### Statistics

function mean(d::InverseWishart)
    df = d.df
    p = dim(d)
    r = df - (p + 1)
    if r > 0.0
        return Matrix(d.Ψ) * (1.0 / r)
    else
        error("mean only defined for df > p + 1")
    end
end

mode(d::InverseWishart) = d.Ψ * inv(d.df + dim(d) + 1.0)

#  https://en.wikipedia.org/wiki/Inverse-Wishart_distribution#Moments
function cov(d::InverseWishart, i::Integer, j::Integer, k::Integer, l::Integer)
    p, ν, Ψ = (dim(d), d.df, Matrix(d.Ψ))
    ν > p + 3 || throw(ArgumentError("cov only defined for df > dim + 3"))
    inv((ν - p)*(ν - p - 3)*(ν - p - 1)^2)*(2Ψ[i,j]*Ψ[k,l] + (ν-p-1)*(Ψ[i,k]*Ψ[j,l] + Ψ[i,l]*Ψ[k,j]))
end

function var(d::InverseWishart, i::Integer, j::Integer)
    p, ν, Ψ = (dim(d), d.df, Matrix(d.Ψ))
    ν > p + 3 || throw(ArgumentError("var only defined for df > dim + 3"))
    inv((ν - p)*(ν - p - 3)*(ν - p - 1)^2)*(ν - p + 1)*Ψ[i,j]^2 + (ν - p - 1)*Ψ[i,i]*Ψ[j,j]
end

#### Evaluation

function _logpdf(d::InverseWishart, X::AbstractMatrix)
    p = dim(d)
    df = d.df
    Xcf = cholesky(X)
    # we use the fact: tr(Ψ * inv(X)) = tr(inv(X) * Ψ) = tr(X \ Ψ)
    Ψ = Matrix(d.Ψ)
    -0.5 * ((df + p + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end


#### Sampling

_rand!(rng::AbstractRNG, d::InverseWishart, A::AbstractMatrix) =
    (A .= inv(cholesky!(_rand!(rng, Wishart(d.df, inv(d.Ψ)), A))))

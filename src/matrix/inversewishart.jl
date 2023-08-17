"""
    InverseWishart(ν, Ψ)
```julia
ν::Real           degrees of freedom (greater than p - 1)
Ψ::AbstractPDMat  p x p scale matrix
```
The [inverse Wishart distribution](http://en.wikipedia.org/wiki/Inverse-Wishart_distribution)
generalizes the inverse gamma distribution to ``p\\times p`` real, positive definite
matrices ``\\boldsymbol{\\Sigma}``.
If ``\\boldsymbol{\\Sigma}\\sim \\textrm{IW}_p(\\nu,\\boldsymbol{\\Psi})``,
then its probability density function is

```math
f(\\boldsymbol{\\Sigma}; \\nu,\\boldsymbol{\\Psi}) =
\\frac{\\left|\\boldsymbol{\\Psi}\\right|^{\\nu/2}}{2^{\\nu p/2}\\Gamma_p(\\frac{\\nu}{2})} \\left|\\boldsymbol{\\Sigma}\\right|^{-(\\nu+p+1)/2} e^{-\\frac{1}{2}\\operatorname{tr}(\\boldsymbol{\\Psi}\\boldsymbol{\\Sigma}^{-1})}.
```

``\\mathbf{H}\\sim \\textrm{W}_p(\\nu, \\mathbf{S})`` if and only if
``\\mathbf{H}^{-1}\\sim \\textrm{IW}_p(\\nu, \\mathbf{S}^{-1})``.
"""
struct InverseWishart{T<:Real, ST<:AbstractPDMat} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    Ψ::ST     # scale matrix
    logc0::T  # log of normalizing constant
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function InverseWishart(df::T, Ψ::AbstractPDMat{T}) where T<:Real
    p = size(Ψ, 1)
    df > p - 1 || throw(ArgumentError("df should be greater than dim - 1."))
    logc0 = invwishart_logc0(df, Ψ)
    R = Base.promote_eltype(T, logc0)
    prom_Ψ = convert(AbstractArray{R}, Ψ)
    InverseWishart{R, typeof(prom_Ψ)}(R(df), prom_Ψ, R(logc0))
end

function InverseWishart(df::Real, Ψ::AbstractPDMat)
    T = Base.promote_eltype(df, Ψ)
    InverseWishart(T(df), convert(AbstractArray{T}, Ψ))
end

InverseWishart(df::Real, Ψ::Matrix) = InverseWishart(df, PDMat(Ψ))

InverseWishart(df::Real, Ψ::Cholesky) = InverseWishart(df, PDMat(Ψ))

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

show(io::IO, d::InverseWishart) = show_multline(io, d, [(:df, d.df), (:Ψ, Matrix(d.Ψ))])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{InverseWishart{T}}, d::InverseWishart) where T<:Real
    P = convert(AbstractArray{T}, d.Ψ)
    InverseWishart{T, typeof(P)}(T(d.df), P, T(d.logc0))
end
Base.convert(::Type{InverseWishart{T}}, d::InverseWishart{T}) where {T<:Real} = d

function convert(::Type{InverseWishart{T}}, df, Ψ::AbstractPDMat, logc0) where T<:Real
    P = convert(AbstractArray{T}, Ψ)
    InverseWishart{T, typeof(P)}(T(df), P, T(logc0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

insupport(::Type{InverseWishart}, X::Matrix) = isposdef(X)
insupport(d::InverseWishart, X::Matrix) = size(X) == size(d) && isposdef(X)

size(d::InverseWishart) = size(d.Ψ)
rank(d::InverseWishart) = rank(d.Ψ)

params(d::InverseWishart) = (d.df, d.Ψ)
@inline partype(d::InverseWishart{T}) where {T<:Real} = T

function mean(d::InverseWishart)
    df = d.df
    p = size(d, 1)
    r = df - (p + 1)
    r > 0.0 || throw(ArgumentError("mean only defined for df > p + 1"))
    return Matrix(d.Ψ) * (1.0 / r)
end

mode(d::InverseWishart) = d.Ψ * inv(d.df + size(d, 1) + 1.0)

#  https://en.wikipedia.org/wiki/Inverse-Wishart_distribution#Moments
function cov(d::InverseWishart, i::Integer, j::Integer, k::Integer, l::Integer)
    p, ν, Ψ = (size(d, 1), d.df, Matrix(d.Ψ))
    ν > p + 3 || throw(ArgumentError("cov only defined for df > dim + 3"))
    inv((ν - p)*(ν - p - 3)*(ν - p - 1)^2)*(2Ψ[i,j]*Ψ[k,l] + (ν-p-1)*(Ψ[i,k]*Ψ[j,l] + Ψ[i,l]*Ψ[k,j]))
end

function var(d::InverseWishart, i::Integer, j::Integer)
    p, ν, Ψ = (size(d, 1), d.df, Matrix(d.Ψ))
    ν > p + 3 || throw(ArgumentError("var only defined for df > dim + 3"))
    inv((ν - p)*(ν - p - 3)*(ν - p - 1)^2)*((ν - p + 1)*Ψ[i,j]^2 + (ν - p - 1)*Ψ[i,i]*Ψ[j,j])
end

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function invwishart_logc0(df::Real, Ψ::AbstractPDMat)
    h_df = df / 2
    p = size(Ψ, 1)
    -h_df * (p * typeof(df)(logtwo) - logdet(Ψ)) - logmvgamma(p, h_df)
end

function logkernel(d::InverseWishart, X::AbstractMatrix)
    p = size(d, 1)
    df = d.df
    Xcf = cholesky(X)
    # we use the fact: tr(Ψ * inv(X)) = tr(inv(X) * Ψ) = tr(X \ Ψ)
    Ψ = Matrix(d.Ψ)
    -0.5 * ((df + p + 1) * logdet(Xcf) + tr(Xcf \ Ψ))
end

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

_rand!(rng::AbstractRNG, d::InverseWishart, A::AbstractMatrix) =
    (A .= inv(cholesky!(_rand!(rng, Wishart(d.df, inv(d.Ψ)), A))))

#  -----------------------------------------------------------------------------
#  Test utils
#  -----------------------------------------------------------------------------

function _univariate(d::InverseWishart)
    check_univariate(d)
    ν, Ψ = params(d)
    α = ν / 2
    β = Matrix(Ψ)[1] / 2
    return InverseGamma(α, β)
end

function _rand_params(::Type{InverseWishart}, elty, n::Int, p::Int)
    n == p || throw(ArgumentError("dims must be equal for InverseWishart"))
    ν = elty( n + 3 + abs(10randn()) )
    Ψ = (X = 2rand(elty, n, n) .- 1 ; X * X')
    return ν, Ψ
end

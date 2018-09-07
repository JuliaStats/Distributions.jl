"""
    InverseWishart(nu, P)

The [Inverse Wishart distribution](http://en.wikipedia.org/wiki/Inverse-Wishart_distribution)
is usually used as the conjugate prior for the covariance matrix of a multivariate normal
distribution, which is characterized by a degree of freedom ν, and a base matrix Φ.
"""
struct InverseWishart{T<:Real, ST<:AbstractPDMat} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    Ψ::ST           # scale matrix
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
params(d::InverseWishart) = (d.df, d.Ψ, d.c0)
@inline partype(d::InverseWishart{T}) where {T<:Real} = T

### Conversion
function convert(::Type{InverseWishart{T}}, d::InverseWishart) where T<:Real
    P = Wishart{T}(d.Ψ)
    InverseWishart{T, typeof(P)}(T(d.df), P, T(d.c0))
end
function convert(::Type{InverseWishart{T}}, df, Ψ::AbstractPDMat, c0) where T<:Real
    P = Wishart{T}(Ψ)
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

rand(d::InverseWishart) = inv(cholesky!(rand(Wishart(d.df, inv(d.Ψ)))))

function _rand!(d::InverseWishart, X::AbstractArray{M}) where M<:Matrix
    wd = Wishart(d.df, inv(d.Ψ))
    for i in 1:length(X)
        X[i] = inv(cholesky!(rand(wd)))
    end
    return X
end

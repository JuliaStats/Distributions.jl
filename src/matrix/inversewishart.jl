# Inverse Wishart distribution
#
#   following Wikipedia parametrization
#

immutable InverseWishart{ST<:AbstractPDMat} <: ContinuousMatrixDistribution
    df::Float64     # degree of freedom
    Ψ::ST           # scale matrix
    c0::Float64     # log of normalizing constant
end

#### Constructors

function InverseWishart{ST<:AbstractPDMat}(df::Real, Ψ::ST)
    p = dim(Ψ)
    df > p - 1 || error("df should be greater than dim - 1.")
    InverseWishart{ST}(df, Ψ, _invwishart_c0(df, Ψ))
end

InverseWishart(df::Real, Ψ::Matrix{Float64}) = InverseWishart(df, PDMat(Ψ))

InverseWishart(df::Real, Ψ::Cholesky) = InverseWishart(df, PDMat(Ψ))

function _invwishart_c0(df::Real, Ψ::AbstractPDMat)
    h_df = df / 2
    p = dim(Ψ)
    h_df * (p * logtwo - logdet(Ψ)) + lpgamma(p, h_df)
end


#### Properties

insupport(::Type{InverseWishart}, X::Matrix{Float64}) = isposdef(X)
insupport(d::InverseWishart, X::Matrix{Float64}) = size(X) == size(d) && isposdef(X)

dim(d::InverseWishart) = dim(d.Ψ)
size(d::InverseWishart) = (p = dim(d); (p, p))


#### Show

show(io::IO, d::InverseWishart) = show_multline(io, d, [(:df, d.df), (:Ψ, full(d.Ψ))])


#### Statistics

function mean(d::InverseWishart)
    df = d.df
    p = dim(d)
    r = df - (p + 1)
    if r > 0.0
        return full(d.Ψ) * (1.0 / r)
    else
        error("mean only defined for df > p + 1")
    end
end

mode(d::InverseWishart) = d.Ψ * inv(d.df + dim(d) + 1.0)


#### Evaluation

function _logpdf(d::InverseWishart, X::DenseMatrix{Float64})
    p = dim(d)
    df = d.df
    Xcf = cholfact(X)
    # we use the fact: trace(Ψ * inv(X)) = trace(inv(X) * Ψ) = trace(X \ Ψ)
    Ψ = full(d.Ψ)
    -0.5 * ((df + p + 1) * logdet(Xcf) + trace(Xcf \ Ψ)) - d.c0
end

@compat _logpdf{T<:Real}(d::InverseWishart, X::DenseMatrix{T}) = _logpdf(d, Float64(X))


#### Sampling

rand(d::InverseWishart) = inv(rand(Wishart(d.df, inv(d.Ψ))))

function _rand!{M<:Matrix}(d::InverseWishart, X::AbstractArray{M})
    wd = Wishart(d.df, inv(d.Ψ))
    for i in 1:length(X)
        X[i] = inv(rand(wd))
    end
    return X
end

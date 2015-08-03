# Wishart distribution
#
#   following the Wikipedia parameterization
#

immutable Wishart{ST<:AbstractPDMat} <: ContinuousMatrixDistribution
    df::Float64     # degree of freedom
    S::ST           # the scale matrix
    c0::Float64     # the logarithm of normalizing constant in pdf
end

#### Constructors

function Wishart{ST<:AbstractPDMat}(df::Real, S::ST)
    p = dim(S)
    df > p - 1 || error("df should be greater than dim - 1.")
    Wishart{ST}(df, S, _wishart_c0(df, S))
end

Wishart(df::Real, S::Matrix{Float64}) = Wishart(df, PDMat(S))

Wishart(df::Real, S::Cholesky) = Wishart(df, PDMat(S))

function _wishart_c0(df::Float64, S::AbstractPDMat)
    h_df = df / 2
    p = dim(S)
    h_df * (logdet(S) + p * logtwo) + logmvgamma(p, h_df)
end


#### Properties

insupport(::Type{Wishart}, X::Matrix{Float64}) = isposdef(X)
insupport(d::Wishart, X::Matrix{Float64}) = size(X) == size(d) && isposdef(X)

dim(d::Wishart) = dim(d.S)
size(d::Wishart) = (p = dim(d); (p, p))


#### Show

show(io::IO, d::Wishart) = show_multline(io, d, [(:df, d.df), (:S, full(d.S))])


#### Statistics

mean(d::Wishart) = d.df * full(d.S)

function mode(d::Wishart)
    r = d.df - dim(d) - 1.0
    if r > 0.0
        return full(d.S) * r
    else
        error("mode is only defined when df > p + 1")
    end
end

function meanlogdet(d::Wishart)
    p = dim(d)
    df = d.df
    v = logdet(d.S) + p * logtwo
    for i = 1:p
        v += digamma(0.5 * (df - (i - 1)))
    end
    return v
end

function entropy(d::Wishart)
    p = dim(d)
    df = d.df
    d.c0 - 0.5 * (df - p - 1) * meanlogdet(d) + 0.5 * df * p
end


#### Evaluation

function _logpdf(d::Wishart, X::DenseMatrix{Float64})
    df = d.df
    p = dim(d)
    Xcf = cholfact(X)
    0.5 * ((df - (p + 1)) * logdet(Xcf) - trace(d.S \ X)) - d.c0
end

_logpdf{T<:Real}(d::Wishart, X::DenseMatrix{T}) = _logpdf(d, convert(Matrix{Float64}, X))

#### Sampling

function rand(d::Wishart)
    Z = unwhiten!(d.S, _wishart_genA(dim(d), d.df))
    A_mul_Bt(Z, Z)
end

function _wishart_genA(p::Int, df::Float64)
    # Generate the matrix A in the Bartlett decomposition
    #
    #   A is a lower triangular matrix, with
    #
    #       A(i, j) ~ sqrt of Chisq(df - i + 1) when i == j
    #               ~ Normal()                  when i > j
    #
    A = zeros(p, p)
    for i = 1:p
        @inbounds A[i,i] = sqrt(rand(Chisq(df - i + 1.0)))
    end
    for j = 1:p-1, i = j+1:p
        @inbounds A[i,j] = randn()
    end
    return A
end

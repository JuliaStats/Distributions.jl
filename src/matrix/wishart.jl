# Wishart distribution
#
#   following the Wikipedia parameterization
#

immutable Wishart{ST<:AbstractPDMat, T<:Real} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    S::ST           # the scale matrix
    c0::T     # the logarithm of normalizing constant in pdf
end

#### Constructors

function Wishart{ST <: AbstractPDMat}(df::Real, S::ST)
    p = dim(S)
    df > p - 1 || error("dpf should be greater than dim - 1.")
    c0 = _wishart_c0(df, S)
    prom_df, prom_c0 = promote(df, c0)
    T = typeof(prom_df)
    Wishart{ST, T}(prom_df, S, prom_c0)
end

Wishart(df::Real, S::Matrix) = Wishart(df, PDMat(S))

Wishart(df::Real, S::Cholesky) = Wishart(df, PDMat(S))

function _wishart_c0(df::Real, S::AbstractPDMat)
    h_df = df / 2
    p = dim(S)
    h_df * (logdet(S) + p * logtwo) + logmvgamma(p, h_df)
end


#### Properties

insupport(::Type{Wishart}, X::Matrix) = isposdef(X)
insupport(d::Wishart, X::Matrix) = size(X) == size(d) && isposdef(X)

dim(d::Wishart) = dim(d.S)
size(d::Wishart) = (p = dim(d); (p, p))
params(d::Wishart) = (d.df, d.S, d.c0)

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

function _logpdf(d::Wishart, X::AbstractMatrix)
    df = d.df
    p = dim(d)
    Xcf = cholfact(X)
    0.5 * ((df - (p + 1)) * logdet(Xcf) - trace(d.S \ X)) - d.c0
end

#### Sampling

function rand(d::Wishart)
    Z = unwhiten!(d.S, _wishart_genA(dim(d), d.df))
    A_mul_Bt(Z, Z)
end

function _wishart_genA(p::Int, df::Real)
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


##### Generic methods #####

"""
    size(d::MatrixDistribution)

Return the size of each sample from distribution `d`.
"""
Base.size(d::MatrixDistribution)

size(d::MatrixDistribution, i) = size(d)[i]

"""
    length(d::MatrixDistribution)

The length (*i.e* number of elements) of each sample from the distribution `d`.
"""
Base.length(d::MatrixDistribution)

"""
    rank(d::MatrixDistribution)

The rank of each sample from the distribution `d`.
"""
LinearAlgebra.rank(d::MatrixDistribution)

"""
    inv(d::MatrixDistribution)

If known, returns a `MatrixDistribution` instance representing the
distribution of inv(X), where X is a random matrix with distribution `d`.
"""
Base.inv(d::MatrixDistribution)

"""
    mean(d::MatrixDistribution)

Return the mean matrix of `d`.
"""
mean(d::MatrixDistribution)

"""
    var(d::MatrixDistribution)

Compute the matrix of element-wise variances for distribution `d`.
"""
var(d::MatrixDistribution) = ((n, p) = size(d); [var(d, i, j) for i in 1:n, j in 1:p])

"""
    cov(d::MatrixDistribution)

Compute the covariance matrix for `vec(X)`, where `X` is a random matrix with distribution `d`.
"""
function cov(d::MatrixDistribution)
    M = length(d)
    V = zeros(partype(d), M, M)
    iter = CartesianIndices(size(d))
    for el1 = 1:M
        for el2 = 1:el1
            i, j = Tuple(iter[el1])
            k, l = Tuple(iter[el2])
            V[el1, el2] = cov(d, i, j, k, l)
        end
    end
    return V + tril(V, -1)'
end
cov(d::MatrixDistribution, ::Val{true}) = cov(d)

"""
    cov(d::MatrixDistribution, flattened = Val(false))

Compute the 4-dimensional array whose `(i, j, k, l)` element is `cov(X[i,j], X[k, l])`.
"""
function cov(d::MatrixDistribution, ::Val{false})
    n, p = size(d)
    [cov(d, i, j, k, l) for i in 1:n, j in 1:p, k in 1:n, l in 1:p]
end

# pdf & logpdf

# TODO: Remove or restrict - this causes many ambiguity errors...
_logpdf(d::MatrixDistribution, X::AbstractMatrix{<:Real}) = logkernel(d, X) + d.logc0

#  for testing
is_univariate(d::MatrixDistribution) = size(d) == (1, 1)
check_univariate(d::MatrixDistribution) = is_univariate(d) || throw(ArgumentError("not 1 x 1"))

##### Specific distributions #####

for fname in ["wishart.jl", "inversewishart.jl", "matrixnormal.jl",
              "matrixtdist.jl", "matrixbeta.jl", "matrixfdist.jl", "lkj.jl"]
    include(joinpath("matrix", fname))
end

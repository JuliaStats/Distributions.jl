"""
    MatrixGaussian(M, Σ)
```julia
M::AbstractMatrix  m x n mean
Σ::AbstractPDMat   nm x nm covariance
```
The [matrix Gaussian distribution](https://arxiv.org/pdf/1804.11010.pdf) generalizes the multivariate normal distribution to ``m\\times n`` real matrices ``\\mathbf{X}``.
If ``\\mathbf{X}\\sim {\\cal N}(\\mathbf{M}, \\mathbf{\\Sigma})``, then its
probability density function is

```math
f(\\mathbf{X};\\mathbf{M}, \\mathbf{\\Sigma}) = \\frac{\\exp\\left( -\\frac{1}{2} \\, \\left[ \\textbf{vec}(\\mathbf{X} - \\mathbf{M})^{\\rm{T}} \\mathbf{\\Sigma}^{-1} \\textbf{vec}(\\mathbf{X} - \\mathbf{M}) \\right] \\right)}{(2\\pi)^{mn/2} |\\mathbf{\\Sigma}|^{1/2}}.
```

"""
struct MatrixGaussian <: ContinuousMatrixDistribution
    m::Integer
    n::Integer
    N::MvNormal
end

function MatrixGaussian(M::AbstractMatrix{T}, Σ::AbstractPDMat{T}) where T <: Real
    m, n = size(M)
    (n*m, n*m) == size(Σ) || throw(ArgumentError("$(m) by $(n) mean matrix M requires covariance of size $(n*m) by $(n*m), not $(size(Σ))"))
    MatrixGaussian(size(M)..., MvNormal(vec(M), Σ))
end
MatrixGaussian(m::Integer, n::Integer) = MatrixGaussian(m, n, MvNormal(zeros(m*n), Matrix(1.0I, m*n, m*n)))

show(io::IO, d::MatrixGaussian) = show_multline(io, d, [(:m, d.m), (:n, d.n), (:N, d.N)])


#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::MatrixGaussian) = (d.m, d.n)

rank(d::MatrixGaussian) = minimum(size(d))

insupport(d::MatrixGaussian, X::AbstractMatrix) = insupport(d.N, vec(X))

mean(d::MatrixGaussian) = reshape(mean(d.N), size(d))

mode(d::MatrixGaussian) = mean(d)

cov(d::MatrixGaussian) = cov(d.N)

var(d::MatrixGaussian) = reshape(var(d.N), size(d))

params(d::MatrixGaussian) = (d.m, d.n, params(d.N)...)


#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

_logpdf(d::MatrixGaussian, A::AbstractMatrix) = logpdf(d.N, vec(A))

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

#  https://en.wikipedia.org/wiki/Matrix_normal_distribution#Drawing_values_from_the_distribution


function _rand!(rng::AbstractRNG, d::MatrixGaussian, Y::AbstractMatrix)
    Y .= reshape(rand(rng, d.N), d.m, d.n)
end

#  -----------------------------------------------------------------------------
#  Transformation
#  -----------------------------------------------------------------------------

vec(d::MatrixGaussian) = d.N

using Distributions
using Test

import Distributions: GenericMvTDist
import PDMats: PDMat

# Set location vector mu and scale matrix Sigma as in
# Hofert M. On Sampling from the Multivariate t Distribution. The R Journal
mu = [1., 2]
Sigma = [4. 2; 2 3]

# LogPDF evaluation for varying degrees of freedom df
# Julia's output is compared to R's corresponding values obtained via R's mvtnorm package
# R code exemplifying how the R values (rvalues) were obtained:
# options(digits=20)
# library("mvtnorm")
# mu <- 1:2
# Sigma <- matrix(c(4, 2, 2, 3), ncol=2)
# dmvt(c(-2., 3.), delta=mu, sigma=Sigma, df=1)
rvalues = [-5.6561739738159975133,
           -5.4874952805811396672,
           -5.4441948098568158088,
           -5.432461875138580254,
           -5.4585441614404803801]
df = [1., 2, 3, 5, 10]
for i = 1:length(df)
    d = MvTDist(df[i], mu, Sigma)
    @test isapprox(logpdf(d, [-2., 3]), rvalues[i], atol=1.0e-8)
    dd = typeof(d)(params(d)...)
    @test d.df == dd.df
    @test Vector(d.μ) == Vector(dd.μ)
    @test Matrix(d.Σ) == Matrix(dd.Σ)
end

# test constructors for mixed inputs:
@test typeof(GenericMvTDist(1, Vector{Float32}(mu), PDMat(Sigma))) == typeof(GenericMvTDist(1., mu, PDMat(Sigma)))

@test typeof(GenericMvTDist(1, mu, PDMat(Array{Float32}(Sigma)))) == typeof(GenericMvTDist(1., mu, PDMat(Sigma)))

d = GenericMvTDist(1, Array{Float32}(mu), PDMat(Array{Float32}(Sigma)))
@test typeof(convert(GenericMvTDist{Float64}, d)) == typeof(GenericMvTDist(1, mu, PDMat(Sigma)))
@test typeof(convert(GenericMvTDist{Float64}, d.df, d.dim, d.zeromean, d.μ, d.Σ)) == typeof(GenericMvTDist(1, mu, PDMat(Sigma)))
@test partype(d) == Float32

@test size(rand(MvTDist(1., mu, Sigma))) == (2,)
@test size(rand(MvTDist(1., mu, Sigma), 10)) == (2,10)

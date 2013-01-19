using Distributions

require("nearequal.jl")
require("test.jl")

# n probability points, i.e. the midpoints of the intervals [0, 1/n],...,[1-1/n, 1]
probpts(n::Int) = ((1:n) - 0.5)/n  
pp  = float(probpts(1000))              # convert from a Range{Float64}
lpp = log(pp)

tol = sqrt(eps())

function absdiff{T<:Real}(current::AbstractArray{T}, target::AbstractArray{T})
    @assert all(size(current) == size(target))
    max(abs(current - target))
end

function reldiff{T<:Real}(current::T, target::T)
    abs((current - target)/(bool(target) ? target : 1))
end

function reldiff{T<:Real}(current::AbstractArray{T}, target::AbstractArray{T})
    @assert all(size(current) == size(target))
    max([reldiff(current[i], target[i]) for i in 1:length(target)])
end
    
## Checks on ContinuousDistribution instances
for d in (Beta(), Cauchy(), Chisq(12), Exponential(), Exponential(23.1),
          FDist(2, 21), Gamma(3), Gamma(), Logistic(), logNormal(),
          Normal(), TDist(1), TDist(28), Uniform(), Weibull(2.3))
##    println(d)  # uncomment if an assertion fails
    qq = quantile(d, pp)
    @assert absdiff(cdf(d, qq), pp) < tol
    @assert absdiff(ccdf(d, qq), 1 - pp) < tol
    @assert reldiff(cquantile(d, 1 - pp), qq) < tol
    @assert reldiff(logpdf(d, qq), log(pdf(d, qq))) < tol
    @assert reldiff(logcdf(d, qq), lpp) < tol
    @assert reldiff(logccdf(d, qq), lpp[end:-1:1]) < tol
    @assert reldiff(invlogcdf(d, lpp), qq) < tol
    @assert reldiff(invlogccdf(d, lpp), qq[end:-1:1]) < tol
end

# Additional tests on the Multinomial and Dirichlet constructors
d = Multinomial(1, [0.5, 0.4, 0.1])
d = Multinomial(1, 3)
d = Multinomial(2)
mean(d)
var(d)
@assert insupport(d, [1, 0])
@assert !insupport(d, [1, 1])
@assert insupport(d, [0, 1])
pmf(d, [1, 0])
pmf(d, [1, 1])
pmf(d, [0, 1])
logpmf(d, [1, 0])
logpmf(d, [1, 1])
logpmf(d, [0, 1])
d.n = 10
rand(d)
A = zeros(Int, 2, 10)
rand!(d, A)

d = Dirichlet([1.0, 2.0, 1.0])
d = Dirichlet(3)
mean(d)
var(d)
insupport(d, [0.1, 0.8, 0.1])
insupport(d, [0.1, 0.8, 0.2])
insupport(d, [0.1, 0.8])
pdf(d, [0.1, 0.8, 0.1])
rand(d)
A = zeros(Float64, 10, 3)
rand!(d, A)

d = Categorical([0.25, 0.5, 0.25])
d = Categorical(3)
d = Categorical([0.25, 0.5, 0.25])

@assert !insupport(d, 0)
@assert insupport(d, 1)
@assert insupport(d, 2)
@assert insupport(d, 3)
@assert !insupport(d, 4)

@assert logpmf(d, 1) == log(0.25)
@assert pmf(d, 1) == 0.25

@assert logpmf(d, 2) == log(0.5)
@assert pmf(d, 2) == 0.5

@assert logpmf(d, 0) == -Inf
@assert pmf(d, 0) == 0.0

@assert 1.0 <= rand(d) <= 3.0

A = zeros(Int, 10)
rand!(d, A)
@assert 1.0 <= mean(A) <= 3.0

# Examples of sample()
a = [1, 6, 19]
p = rand(Dirichlet(3))
x = sample(a, p)
@assert x == 1 || x == 6 || x == 19

a = 19.0 * [1.0, 0.0]
x = sample(a)
@assert x == 0.0 || x == 19.0

## Link function tests
const ep = eps()
const oneMeps = 1 - ep
srand(1)

etas = (linspace(-7., 7., 15),  # equal spacing to asymptotic area
        14 * rand(17) - 7,      # random sample from wide uniform dist
        clamp(rand(Normal(0, 4), 17), -7., 7.), # random sample from wide normal dist
        [-7., rand(Normal(0, 4),15), 7.])

## sample linear predictor values for the families in which eta must be positive
etapos = (float64(1:20), rand(Exponential(), 20), rand(Gamma(3), 20), max(ep, rand(Normal(2.), 20)))

## values of mu in the (0,1) interval
mubinom = (rand(100), rand(Beta(1,3), 100),
           [ccall((:rbeta, :libRmath), Float64, (Float64,Float64), 0.1, 3) for i in 1:100],
           [ccall((:rbeta, :libRmath), Float64, (Float64,Float64), 3, 0.1) for i in 1:100])

for ll in (LogitLink(), ProbitLink()#, CloglogLink() # edge cases for CloglogLink are tricky
           , CauchitLink())
#    println(ll)  # Having problems with the edge when eta is very large or very small
#    for i in 1:size(etas,1)
#        println(i)
#        @assert all(isapprox(linkfun(ll, clamp(linkinv(ll, etas[i]), realmin(Float64), 1.-eps())), etas[i]))
#    end
    for mu in mubinom
        mm = clamp(mu, realmin(), oneMeps)
        @assert all(isapprox(linkinv(ll, linkfun(ll, mm)), mm))
    end
end

d = MultivariateNormal(zeros(2), eye(2))
@assert abs(pdf(d, [0, 0]) - 0.159155) < 10e-3
@assert abs(pdf(d, [1, 0]) - 0.0965324) < 10e-3
@assert abs(pdf(d, [1, 1]) - 0.0585498) < 10e-3

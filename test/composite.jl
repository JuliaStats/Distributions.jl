using Base.Test
using Distributions

include("../src/multivariate/composite.jl")
using CompositeDistributions

d1 = Rayleigh(0.3)
d2 = MvNormal([log(0.1),log(0.01)],[0.2,log(0.02)])

d = CompositeDist(ContinuousDistribution[d1, d2])

param = params(d)
@test d1 == Rayleigh(param[1][1])
@test d2 == MvNormal(param[2][1], param[2][2])


# Test basic statistics
l1 = length(d1)
l2 = length(d2)
l = length(d)
@test l1+l2 == l

m1 = mean(d1)
m2 = mean(d2)
m = mean(d)
@test m1 == m[1]
@test m2 == m[2:3]

m1 = mode(d1)
m2 = mode(d2)
m = mode(d)
@test m1 == m[1]
@test m2 == m[2:3]

v1 = var(d1)
v2 = var(d2)
v = var(d)
@test v1 == v[1]
@test v2 == v[2:3]

cv2 = cov(d2)
cv = cov(d)
@test v1 == cv[1,1]
@test all(zeros(2) .== cv[1,2:3])
@test all(zeros(2) .== cv[2:3,1])
@test all(cv2 .== cv[2:3,2:3])

e1 = entropy(d1)
e2 = entropy(d2)
esum = entropy(d)
@test_approx_eq(e1+e2,esum) 


# Now sure how to test these well
r = rand(d)
@test insupport(d,r)

n = 10
r10 = rand(d,n)
@test all(insupport(d,r10))


p1 = logpdf(d1,r[1])
p2 = logpdf(d2,r[2:3])
p = logpdf(d,r)
@test_approx_eq(p1+p2,p) 

p1 = pdf(d1,r[1])
p2 = pdf(d2,r[2:3])
p = pdf(d,r)
@test_approx_eq(p1*p2,p) 

out = zeros(n)
logpdf!(out,d,r10)
for i in 1:n
  @test logpdf(d,r10[:,i]) == out[i]
end



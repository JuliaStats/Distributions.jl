using Distributions
using Base.Test

# Test constructors
@test_throws MethodError TriangularDist()
@test_throws MethodError TriangularDist(3.0)
@test_throws MethodError TriangularDist(4.5, 2.4)
@test_throws ErrorException TriangularDist(4.0, 2.0, 3.0)
@test_throws ErrorException TriangularDist(3.0, 3.0, 3.0)
@test_throws ErrorException TriangularDist(3.0, 5.0, 2.0)
@test_throws ErrorException TriangularDist(3.0, 5.0, 6.0)

# Create two distributions for further tests
# There are two so that all code-paths of some functions
# can be tested.
a1 = 3.0
b1 = 5.0
c1 = 7.0/2.0
d1 = TriangularDist(a1,b1,c1)
a2 = 3.0
b2 = 5.0
c2 = 9.0/2.0
d2 = TriangularDist(a2,b2,c2)

@test isupperbounded(d1)==true
@test isupperbounded(TriangularDist)==true
@test islowerbounded(d1)==true
@test islowerbounded(TriangularDist)==true
@test isbounded(d1)==true
@test isbounded(TriangularDist)==true

@test minimum(d1)==a1
@test maximum(d1)==b1
@test insupport(d1, 3.0)==true
@test insupport(d1, 5.0)==true
@test insupport(d1, 3.5)==true
@test insupport(d1, 3.1)==true
@test insupport(d1, 4.999)==true
@test insupport(d1, 1.0)==false
@test insupport(d1, 5.5)==false

@test_approx_eq mean(d1) 23./6.
@test_approx_eq median(d1) 5.-sqrt(3./2.0)
@test_approx_eq median(d2) 3.+sqrt(3./2.0)
@test_approx_eq mode(d1) 7./2.

@test_approx_eq var(d1) 13/72
@test_approx_eq skewness(d1) 14*sqrt(2/13)/13
@test_approx_eq kurtosis(d1) 12/5-3

# Double check whether this test is correct
@test_approx_eq entropy(d1) 0.5

@test_approx_eq pdf(d1, 1) 0
@test_approx_eq pdf(d1, 3.1) 0.2
@test_approx_eq pdf(d1, 3.5) 1
@test_approx_eq pdf(d1, 4.0) 2/3
@test_approx_eq pdf(d1, 5.0) 0
@test_approx_eq pdf(d1, 5.1) 0

# Not sure these make sense
@test_approx_eq mgf(d1, 2) 3141.7032569285

# Not sure these make sense
@test_approx_eq cf(d1, 1) -0.706340963216717-0.578379106849962im
@test_approx_eq cf(d1, 3) 0.117846739742880-0.400401950797709im
@test_approx_eq cf(d1, 20) 0.00754699480217164+0.00752726705107994im

@test isnan(quantile(d1,-1))
@test isnan(quantile(d1,6))
@test_approx_eq quantile(d1, 0) a1
@test_approx_eq quantile(d1, 1) b1
@test_approx_eq quantile(d1, 0.1) 3+1/sqrt(10)
@test_approx_eq quantile(d1, 0.4) 5-3/sqrt(5)
@test_approx_eq quantile(d1, 0.5) 5-sqrt(3/2)
@test_approx_eq quantile(d1, 0.9) 5-sqrt(3/10)

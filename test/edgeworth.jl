#

using Distributions
using Base.Test

dg = Gamma(1,1)

d_s = EdgeworthSum(dg,10)
d_m = EdgeworthMean(dg,10)
d_z = EdgeworthZ(dg,10)

dg_s = Gamma(10,1)
dg_m = Gamma(10,0.1)
dg_za = Gamma(10,1/std(dg_s))

for i = 0.01:0.01:0.99

    q = quantile(dg_s,i)    
    @test_approx_eq_eps quantile(d_s,i) q 0.02
    @test_approx_eq_eps cquantile(d_s,1-i) q 0.02
    @test_approx_eq_eps cdf(d_s,q) i 0.002
    @test_approx_eq_eps ccdf(d_s,q) 1-i 0.002
    @test_approx_eq_eps pdf(d_s,q) pdf(dg_s,q) 0.005

    q = quantile(dg_m,i)
    @test_approx_eq_eps quantile(d_m,i) q 0.01
    @test_approx_eq_eps cquantile(d_m,1-i) q 0.01
    @test_approx_eq_eps cdf(d_m,q) i 0.002
    @test_approx_eq_eps ccdf(d_m,q) 1-i 0.002
    @test_approx_eq_eps pdf(d_m,q) pdf(dg_m,q) 0.05
    
    q = quantile(dg_za,i) - mean(dg_za)
    @test_approx_eq_eps quantile(d_z,i) q 0.01
    @test_approx_eq_eps cquantile(d_z,1-i) q 0.01
    @test_approx_eq_eps cdf(d_z,q) i 0.002
    @test_approx_eq_eps ccdf(d_z,q) 1-i 0.002
    @test_approx_eq_eps pdf(d_z,q) pdf(dg_za,q+mean(dg_za)) 0.02

    
end




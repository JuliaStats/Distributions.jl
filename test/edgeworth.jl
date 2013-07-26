#

using Distributions
using Base.Test

dg = Gamma(1,1)

d_s = EdgeworthSum(dg,10)
d_m = EdgeworthMean(dg,10)
d_z = EdgeworthZ(dg,10)

dg_s = Gamma(10,1)
dg_m = Gamma(10,0.1)

for i = 0.01:0.01:0.99
    @test_approx_eq quantile(d_s,i) cquantile(d_s,1-i) 
    @test_approx_eq_eps cdf(d_s,quantile(d_s,i))/i 1.0 0.2
    @test_approx_eq_eps ccdf(d_s,cquantile(d_s,i))/i 1.0 0.2

    @test_approx_eq_eps quantile(d_s,i) quantile(dg_s,i) 0.02

    @test_approx_eq quantile(d_m,i) cquantile(d_m,1-i) 
    @test_approx_eq_eps cdf(d_m,quantile(d_m,i))/i 1.0 0.2
    @test_approx_eq_eps ccdf(d_m,cquantile(d_m,i))/i 1.0 0.2

    @test_approx_eq_eps quantile(d_m,i) quantile(dg_m,i) 0.002

    @test_approx_eq quantile(d_z,i) cquantile(d_z,1-i) 
    @test_approx_eq_eps cdf(d_z,quantile(d_z,i))/i 1.0 0.2
    @test_approx_eq_eps ccdf(d_z,cquantile(d_z,i))/i 1.0 0.2

    @test_approx_eq_eps quantile(d_z,i) (quantile(dg_s,i)-mean(dg_s))/std(dg_s) 0.007
end




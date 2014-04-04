using Distributions
using RmathDist

import Distributions: zval

distlist = [
            Normal(0,1),
            Normal(1e10,1e10),
            Normal(1e-10,1e-10)
            ]

for d in distlist
    for x in [-logspace(-300,300,1_000_000),logspace(-300,300,1_000_000)]
        z = zval(d,x)

        test_rmath(pdf,d,x)
        if !(1e154 < abs(z) < 1e155) 
            # dist.jl overflow: only a small gap, so probably not worth worrying about
            test_rmath(logpdf,d,x)
        end

        if !(37 < abs(z) < 39) # rmath underflow
            test_rmath(cdf,d,x,1e5)
            test_rmath(ccdf,d,x,1e5)
        end
        if !(1e154 < abs(z) < 1e155) # rmath overflow
            test_rmath(logcdf,d,x,1e5)
            test_rmath(logccdf,d,x,1e5)
        end
    end

    for p in logspace(-300,0,1_000_000)
        test_rmath(quantile,d,p,1e5)
        test_rmath(cquantile,d,p,1e5)
    end

    for lp in -logspace(-300,300,1_000_000)
        test_rmath(invlogcdf,d,lp)
        test_rmath(invlogccdf,d,lp)
    end
end

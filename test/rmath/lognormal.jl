using Distributions
using RmathDist

import Distributions: zval

distlist = [
            LogNormal(0,1),
            LogNormal(1e10,1e10),
            LogNormal(1e-10,1e-10)
            ]

for d in distlist
    for x in [logspace(-300,300,1_000_000)]
        z = zval(Normal(d.meanlog,d.sdlog),log(x))

        if x <= 1e298 # rmath underflow
            test_rmath(pdf,d,x,1e4)
            test_rmath(logpdf,d,x,1e5)
        end        
        if !(37 < abs(z) < 39) # rmath underflow
            test_rmath(cdf,d,x,1e5) 
            test_rmath(ccdf,d,x,1e5)
        end

        test_rmath(logcdf,d,x,1e5)
        test_rmath(logccdf,d,x,1e5)
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

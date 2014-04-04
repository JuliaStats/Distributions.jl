using Distributions
using RmathDist

distlist = [
            Uniform(0,1),
            Uniform(1e2,1e10)
            ]
    

for d in distlist
    for x in [logspace(-300,300,1_000_000)]

        test_rmath(pdf,d,x)
        test_rmath(logpdf,d,x)
        
        test_rmath(cdf,d,x)
        test_rmath(ccdf,d,x)
        test_rmath(logcdf,d,x)
        test_rmath(logccdf,d,x)
    end

    for p in logspace(-300,0,1_000_000)
        test_rmath(quantile,d,p)
        test_rmath(cquantile,d,p)
    end

    for lp in -logspace(-300,300,1_000_000)
        test_rmath(invlogcdf,d,lp)
        test_rmath(invlogccdf,d,lp)
    end
end

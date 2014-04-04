using Distributions
using RmathDist

distlist = [
            Logistic(0,1),
            Logistic(1e10,1e10),
            Logistic(1e-10,1e-10)
            ]

for d in distlist
    for x in [-logspace(-300,300,1_000_000),logspace(-300,300,1_000_000)]
        test_rmath(pdf,d,x)
        test_rmath(logpdf,d,x)

        test_rmath(cdf,d,x)
        test_rmath(ccdf,d,x)
        test_rmath(logcdf,d,x)
        test_rmath(logccdf,d,x)
    end

    for p in logspace(-300,0,1_000_000)
        test_rmath(quantile,d,p,1e5)
        test_rmath(cquantile,d,p,1e5)
    end

    for lp in -logspace(-300,300,1_000_000)
        test_rmath(invlogcdf,d,lp,1e5)
        test_rmath(invlogccdf,d,lp,1e5)
    end
end

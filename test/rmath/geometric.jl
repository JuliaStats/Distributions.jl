using Distributions
using RmathDist

distlist = [
            Geometric(0.5),
            Geometric(0.999999999),
            Geometric(1e-10)
            ]
    

for d in distlist
    for x in [0.0:20.0,linspace(21.0,float(typemax(Int32)),1_000_000)]
        x = round(Int,x)

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

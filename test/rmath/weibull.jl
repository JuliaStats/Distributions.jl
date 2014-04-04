using Distributions
using RmathDist

distlist = [
            Weibull(1,1),
            Weibull(1,1e10),
            Weibull(1,1e-10),
            Weibull(1e3,1),
            Weibull(1e3,1e10),
            Weibull(1e3,1e-10),
            Weibull(1e-2,1),
            Weibull(1e-2,1e10),
            # Weibull(1e-2,1e-10) rmath not so good here
            ]
    

for d in distlist
    for x in [logspace(-300,300,1_000_000)]

        if d.shape != 1e3 || (0.5 < x/d.scale < 1.8) # rmath overflow/NaN
            test_rmath(pdf,d,x,1e4)
            if x < 1e290 # dist.jl overflow                            
                test_rmath(logpdf,d,x,1e5)
            end
        end

        test_rmath(cdf,d,x)
        test_rmath(ccdf,d,x)
        test_rmath(logcdf,d,x)
        test_rmath(logccdf,d,x)
    end

    for p in logspace(-300,0,1_000_000)
        test_rmath(quantile,d,p,1e4)
        test_rmath(cquantile,d,p,1e4)
    end

    for lp in -logspace(-300,300,1_000_000)
        test_rmath(invlogcdf,d,lp,1e4)
        test_rmath(invlogccdf,d,lp,1e4)
    end
end

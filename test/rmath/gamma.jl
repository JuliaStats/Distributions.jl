using Distributions
using RmathDist

distlist = [
            Gamma(1.0,1.0),
            Gamma(10.0,1.0),
            Gamma(11.0,1.0),
            Gamma(1e3,1.0),
            Gamma(0.1,1.0),
            ]

for d in distlist
    for x in logspace(-300,300,1_000_000)
        test_rmath(pdf,d,x,1e4)
        test_rmath(logpdf,d,x,1e4)
    end
end

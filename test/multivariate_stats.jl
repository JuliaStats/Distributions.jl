using Distributions
using Test


const n_samples = 5_000_001

mu = [1.0, 2.0, 3.0]
C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
h = mu
J = C


for d in [
    Dirichlet(3, 2.0), 
    Dirichlet([2.0, 1.0, 3.0]), 
    IsoNormal(mu, 2.0), 
    DiagNormal(mu, [1.5, 2.0, 2.5]), 
    MvNormal(mu, C), 
    IsoNormalCanon(h, 2.0), 
    DiagNormalCanon(h, [1.5, 2.0, 1.2]), 
    MvNormalCanon(h, J)]

    println(d)
    dmean = mean(d)
    dcov = cov(d)
    dent = entropy(d)

    x = rand(d, n_samples)
    xmean = vec(mean(x, 2))
    z = x .- xmean
    xcov = (z * z') * (1 / n_samples)

    lp = logpdf(d, x)
    xent = -mean(lp)

    println("expected mean  = $dmean")
    println("empirical mean = $xmean")
    println("--> abs.dev = $(maximum(abs(dmean - xmean)))")

    println("expected cov  = $dcov")
    println("empirical cov = $xcov")
    println("--> abs.dev = $(maximum(abs(dcov - xcov)))")

    println("expected entropy = $dent")
    println("empirical entropy = $xent")
    println("--> abs.dev = $(abs(dent - xent))")

    println()
end

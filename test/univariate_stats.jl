# Comparison of empirical stats with expected stats

using Distributions
using Base.Test

const n_samples = 5_000_001

macro check_deviation(title, v, vhat)    
    quote
        abs_dev = abs($v - $vhat)
        if $v != 0.
            rel_dev = abs_dev / abs($v)
            @printf "    %-8s: expect = %10.3e emp = %10.3e  |  abs.dev = %.2e rel.dev = %.2e\n" $title $v $vhat abs_dev rel_dev
        else
            @printf "    %-8s: expect = %10.3e emp = %10.3e  |  abs.dev = %.2e rel.dev = n/a\n" $title $v $vhat abs_dev
        end
    end
end

macro ignore_methoderror(ex)
    quote
        try
            $(esc(ex))
        catch err
            if !isa(err,MethodError)
                rethrow(err)
            end
        end
    end
end


for d in [Arcsine(),
          Bernoulli(0.1),
          Bernoulli(0.5),
          Bernoulli(0.9),
          Beta(2.0, 2.0),
          Beta(3.0, 4.0),
          Beta(17.0, 13.0),
          # BetaPrime(3.0, 3.0),
          # BetaPrime(3.0, 5.0),
          # BetaPrime(5.0, 3.0),
          Binomial(1, 0.5),
          Binomial(100, 0.1),
          Binomial(100, 0.9),
          Categorical([0.1, 0.9]),
          Categorical([0.5, 0.5]),
          Categorical([0.9, 0.1]),
          Cauchy(0.0, 1.0),
          Cauchy(10.0, 1.0),
          Cauchy(0.0, 10.0),
          Chi(12),
          Chisq(8),
          Chisq(12.0),
          Chisq(20.0),
          # Cosine(),
          DiscreteUniform(0, 3),
          DiscreteUniform(2.0, 5.0),
          # Empirical(),
          Erlang(1),
          Erlang(17.0),
          Exponential(1.0),
          Exponential(5.1),
          FDist(9, 9),
          FDist(9, 21),
          FDist(21, 9),
          Gamma(3.0, 2.0),
          Gamma(2.0, 3.0),
          Gamma(3.0, 3.0),
          Geometric(0.1),
          Geometric(0.5),
          Geometric(0.9),
          Gumbel(3.0, 5.0),
          Gumbel(5, 3),
          # HyperGeometric(1.0, 1.0, 1.0),
          # HyperGeometric(2.0, 2.0, 2.0),
          # HyperGeometric(3.0, 2.0, 2.0),
          # HyperGeometric(2.0, 3.0, 2.0),
          # HyperGeometric(2.0, 2.0, 3.0),
          InverseGaussian(1.0,1.0),
          InverseGaussian(2.0,7.0),
          InvertedGamma(1.0,1.0),
          InvertedGamma(2.0,3.0),
          Laplace(0.0, 1.0),
          Laplace(10.0, 1.0),
          Laplace(0.0, 10.0),
          Levy(0.0, 1.0),
          Levy(2.0, 8.0),
          Levy(3.0, 3.0),
          Logistic(0.0, 1.0),
          Logistic(10.0, 1.0),
          Logistic(0.0, 10.0),
          LogNormal(0.0, 1.0),
          LogNormal(10.0, 1.0),
          LogNormal(0.0, 10.0),
          # NegativeBinomial(),
          # NegativeBinomial(5, 0.6),
          NoncentralBeta(2,2,0),
          NoncentralBeta(2,6,5),
          NoncentralChisq(2,2),
          NoncentralChisq(2,5),
          NoncentralF(2,2,2),
          NoncentralF(8,10,5),
          # NoncentralT(2,2),
          NoncentralT(10,2),
          Normal(0.0, 1.0),
          Normal(-1.0, 10.0),
          Normal(1.0, 10.0),
          # Pareto(),
          Poisson(2.0),
          Poisson(10.0),
          Poisson(51.0),
          Rayleigh(1.0),
          Rayleigh(5.0),
          Rayleigh(10.0),
          # Skellam(10.0, 2.0), # Entropy wrong
          # TDist(1), # Entropy wrong
          # TDist(28), # Entropy wrong
          Triangular(3.0, 1.0),
          Triangular(3.0, 2.0),
          Triangular(10.0, 10.0),
          Truncated(Normal(0, 1), -3, 3),
          # Truncated(Normal(-100, 1), 0, 1),
          Truncated(Normal(27, 3), 0, Inf),
          Uniform(0.0, 1.0),
          Uniform(3.0, 17.0),
          Uniform(3.0, 3.1),
          Weibull(2.3),
          Weibull(23.0),
          Weibull(230.0)]

    x = rand(d, n_samples)


    println(d)
    local mu
    @ignore_methoderror begin
        mu, mu_hat = mean(d), mean(x)
        if isfinite(mu)
            # empirical mean should be close to theoretical value
            @check_deviation "mean" mu mu_hat        
        end
    end

    @ignore_methoderror begin 
        m, m_hat = median(d), median(x)
        @assert insupport(d, m_hat)
        @check_deviation "median" m m_hat
    end
    

    @ignore_methoderror begin 
        sigma2, sigma2_hat = var(d), var(x)
        # empirical variance should be close to theoretical value
        if isfinite(mu) && isfinite(sigma2)       
            @check_deviation "variance" sigma2 sigma2_hat
        end
    end

    @ignore_methoderror begin 
        sk, sk_hat = skewness(d), skewness(x)
        # empirical skewness should be close to theoretical value
        if isfinite(mu) && isfinite(sk) 
            @check_deviation "skewness" sk sk_hat
        end
    end

    @ignore_methoderror begin 
        k, k_hat = kurtosis(d), kurtosis(x)
        # empirical kurtosis should be close to theoretical value
        # very unstable for FDist
        if isfinite(mu) && isfinite(k)
            @check_deviation "kurtosis" k k_hat
        end
    end

    @ignore_methoderror begin 
        ent, ent_hat = entropy(d), -mean(logpdf(d, x))
        # By the Asymptotic Equipartition Property,
        # empirical mean negative log PDF should be close to theoretical value
        if isfinite(ent)
            @check_deviation "entropy" ent ent_hat
        end
    end
        
    # Kolmogorov-Smirnov test
    if isa(d, Truncated) ? isa(d.untruncated, ContinuousDistribution) : isa(d, ContinuousDistribution)
        c = cdf(d,x)
        sort!(c)
        for i = 1:n_samples
            c[i] = c[i]*n_samples - i
        end
        a = max(abs(c))
        for i = 1:n_samples
            c[i] += 1.0
        end
        b = max(abs(c))
        ks = max(a,b)
        kss = ks/n_samples
        ksp = ccdf(Kolmogorov(),ks/sqrt(n_samples))
        @printf "    KS statistic = %10.3e,   p-value = %8.6f \n" kss ksp
    end
    println()
end



# Drop-in replacements for some Rmath functions

pnorm(q, mean=0.0, sd=1.0) = cdf(Normal(mean, sd), q)
qnorm(p, mean=0.0, sd=1.0) = quantile(Normal(mean, sd), p)
rnorm(n, mean=0.0, sd=1.0) = rand(Normal(mean, sd), n)

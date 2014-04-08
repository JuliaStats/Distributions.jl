# Conversions
convert(::Type{Gamma},d::Chisq) = Gamma(0.5*d.df,2.0)

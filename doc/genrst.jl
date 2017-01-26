using Distributions

if VERSION < v"0.5.0-"
    function Markdown.rstinline(io::IO, md::Markdown.Link)
        if ismatch(r":(func|obj|ref|exc|class|const|data):`\.*", md.url)
            Markdown.rstinline(io, md.url)
        else
            Markdown.rstinline(io, "`", md.text, " <", md.url, ">`_")
        end
    end
end
    
function printrst(io,md)
    mdd = md.content[1]
    sigs = shift!(mdd.content)
    
    decl = ".. function:: "*replace(sigs.code, "\n","\n              ")
    body = Markdown.rst(mdd)
    println(io, decl)
    println(io)
    for l in split(body, "\n")
        ismatch(r"^\s*$", l) ? println(io) : println(io, "   ", l)
    end
end

cd(joinpath(dirname(@__FILE__),"source")) do
    for (name,T) in [("Univariate Discrete",DiscreteUnivariateDistribution),
                     ("Univariate Continuous",ContinuousUnivariateDistribution),
                     ]

        fname = replace(lowercase(name),' ','-')
        open("$fname.rst","w") do f
            println(f,".. _$fname:")
            println(f)
            println(f,"$name Distributions")
            println(f,"----------------------------------------------------")
            println(f)
            println(f, ".. DO NOT EDIT: this file is generated from Julia source.")
            println(f)
            
            for D in subtypes(T)
                if isleaftype(D)
                    md = Base.doc(D)
                    if isa(md,Markdown.MD)
                        isa(md.content[1].content[1],Markdown.Code) || error("Incorrect docstring format: $D")

                        printrst(f,md)
                    else
                        warn("$D is not documented.")
                    end
                end
            end
        end
    end
end

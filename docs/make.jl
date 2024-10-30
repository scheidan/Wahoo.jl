using Wahoo
using Documenter

DocMeta.setdocmeta!(Wahoo, :DocTestSetup, :(using Wahoo); recursive=true)

makedocs(;
    modules=[Wahoo],
    authors="Andreas Scheidegger",
    sitename="Wahoo.jl",
    format=Documenter.HTML(;
        canonical="https://scheidan.github.io/Wahoo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/scheidan/Wahoo.jl",
    devbranch="main",
)

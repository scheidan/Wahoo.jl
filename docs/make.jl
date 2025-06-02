using Wahoo
using Documenter

DocMeta.setdocmeta!(Wahoo, :DocTestSetup, :(using Wahoo); recursive=true)

makedocs(;
    modules=[Wahoo],
    authors="Andreas Scheidegger and contributors",
    sitename="Wahoo.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        canonical="https://scheidan.github.io/Wahoo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => "manual.md",
        "Example" => "example.md",
        "API" => "api.md",
        "Internals" => "internal.md",
    ],
         )

deploydocs(;
    repo="github.com/scheidan/Wahoo.jl",
    devbranch="main",
)

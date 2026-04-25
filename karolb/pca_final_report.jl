using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, Statistics

# 1. LOAD DATA
df = CSV.read("for_participants/data/players_quarters_final.csv", DataFrame)
df = filter(row -> row.position != "G", df)
dropmissing!(df, [:last15_hsr, :last15_mean_max_speed, :cumul_hsr])

# 2. FULL PHYSICAL PCA (10 Variables)
phys_vars = [
    "last15_sprints", "last15_hsr", "last15_distance", "last15_mean_max_speed", "last15_peak_speed",
    "cumul_sprints", "cumul_hsr", "cumul_distance", "cumul_mean_max_speed", "cumul_peak_speed"
]

X = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, phys_vars]), dims=1), Matrix{Float64}(df[:, phys_vars]))
pca = fit(PCA, collect(X'); maxoutdim=2)

# Loadings
loadings = projection(pca)
vars_exp = principalvars(pca) ./ tvar(pca)

println("--- PCA VARIANCE SUMMARY ---")
println(DataFrame(Factor=["PC1", "PC2"], Variance_Explained=round.(vars_exp .* 100, digits=2)))

println("\n--- PCA LOADINGS TABLE ---")
loadings_df = DataFrame(Variable=phys_vars, PC1=round.(loadings[:,1], digits=3), PC2=round.(loadings[:,2], digits=3))
println(loadings_df)

# Logic check for interpretation
# PC1 typically overall volume/intensity (all same sign)
# PC2 typically quality/profile (mix of signs)

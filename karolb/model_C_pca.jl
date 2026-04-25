using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics

println("Fitting Model C: Pure PCA (Highest Predictive Power)...")

# 1. DATA PREP
df_main = CSV.read("for_participants/data/players_quarters_final.csv", DataFrame)
df_pass = CSV.read("for_participants/data/player_appearance_pass.csv", DataFrame)
df_press = CSV.read("for_participants/data/player_appearance_behaviour_under_pressure.csv", DataFrame)

get_cp(period, min) = period == "half_1" ? (min <= 15 ? "H1_15" : (min <= 30 ? "H1_30" : "H1_45")) : (min <= 15 ? "H2_15" : (min <= 30 ? "H2_30" : "H2_45"))
df_pass.checkpoint = get_cp.(df_pass.period, df_pass.minute)
df_press.checkpoint = get_cp.(df_press.period, df_press.minute)

pass_agg = combine(groupby(df_pass, [:player_appearance_id, :checkpoint]), nrow => :n_pass)
press_agg = combine(groupby(df_press, [:player_appearance_id, :checkpoint]), nrow => :n_press)

df = leftjoin(df_main, pass_agg, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, press_agg, on=[:player_appearance_id, :checkpoint])

for c in [:n_pass, :n_press]; df[!, c] = coalesce.(df[!, c], 0); end
df = filter(row -> row.position != "G", df)
dropmissing!(df, [:scored_after, :last15_hsr])

# 2. PCA
feats = ["last15_sprints", "last15_hsr", "last15_distance", "cumul_distance", "n_pass", "n_press"]
X = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, feats]), dims=1), Matrix{Float64}(df[:, feats]))
pca = fit(PCA, collect(X'); maxoutdim=4)
pcs = MultivariateStats.transform(pca, collect(X'))'

# 3. REGRESSION
data = DataFrame(pcs, [:PC1, :PC2, :PC3, :PC4])
data.target = Float64.(df.scored_after)
data.position = df.position
data.shots_ot = Float64.(df.cumul_shots_on_target)

pos_c = sum(data.target .== 1); total = length(data.target)
wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in data.target]

m = glm(@formula(target ~ PC1 + PC2 + PC3 + PC4 + position + shots_ot), data, Binomial(), LogitLink(), wts=wts)

println("\n--- MODEL C RESULTS (PURE ACCURACY) ---")
println("AUC: ", round(au_roccurve(data.target, predict(m)), digits=4))
println(coeftable(m))

using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics

println("Fitting Model A: Hybrid (PCA Volume + Direct Tactical)...")

# 1. DATA PREP
df_main = CSV.read("for_participants/data/players_quarters_final.csv", DataFrame)
df_pass = CSV.read("for_participants/data/player_appearance_pass.csv", DataFrame)
df_shot = CSV.read("for_participants/data/player_appearance_shot_limited.csv", DataFrame)

get_cp(period, min) = period == "half_1" ? (min <= 15 ? "H1_15" : (min <= 30 ? "H1_30" : "H1_45")) : (min <= 15 ? "H2_15" : (min <= 30 ? "H2_30" : "H2_45"))
for d in [df_pass, df_shot]; d.checkpoint = get_cp.(d.period, d.minute); end

# Engineering
pass_vol = combine(groupby(df_pass, [:player_appearance_id, :checkpoint]), nrow => :n_pass)
pass_qual = combine(groupby(df_pass, [:player_appearance_id, :checkpoint]), :stage => (x -> sum(x .== "top") / (length(x) + 1)) => :verticality)
shot_qual = combine(groupby(df_shot, [:player_appearance_id, :checkpoint]), :technique => (x -> sum(x .∈ [["volley", "header"]]) / (length(x) + 1)) => :danger_index)

df = leftjoin(df_main, pass_vol, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, pass_qual, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, shot_qual, on=[:player_appearance_id, :checkpoint])

for c in [:n_pass, :verticality, :danger_index]; df[!, c] = coalesce.(df[!, c], 0.0); end
df = filter(row -> row.position != "G", df)
dropmissing!(df, [:scored_after, :last15_hsr])

# 2. PCA ON VOLUME
vol_feats = ["last15_sprints", "last15_hsr", "last15_distance", "n_pass"]
X_vol = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, vol_feats]), dims=1), Matrix{Float64}(df[:, vol_feats]))
pca = fit(PCA, collect(X_vol'); maxoutdim=3)
pcs = MultivariateStats.transform(pca, collect(X_vol'))'

# 3. REGRESSION
data = DataFrame(pcs, [:PC1_Physical, :PC2_Passing, :PC3_WorkRate])
data.verticality = df.verticality
data.danger_index = df.danger_index
data.target = Float64.(df.scored_after)
data.position = df.position
data.shots_ot = Float64.(df.cumul_shots_on_target)

pos_c = sum(data.target .== 1); total = length(data.target)
wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in data.target]

m = glm(@formula(target ~ PC1_Physical + PC2_Passing + PC3_WorkRate + verticality + danger_index + position + shots_ot), data, Binomial(), LogitLink(), wts=wts)

println("\n--- MODEL A RESULTS (STORYTELLING) ---")
println("AUC: ", round(au_roccurve(data.target, predict(m)), digits=4))
println(coeftable(m))

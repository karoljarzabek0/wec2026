using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics, Random

println("Building Improved Final Model...")

# 1. LOAD DATA
df_main = CSV.read("for_participants/data/players_quarters_final.csv", DataFrame)
df_pass = CSV.read("for_participants/data/player_appearance_pass.csv", DataFrame)
df_press = CSV.read("for_participants/data/player_appearance_behaviour_under_pressure.csv", DataFrame)
df_shot = CSV.read("for_participants/data/player_appearance_shot_limited.csv", DataFrame)

# 2. FIXED CHECKPOINT MAPPING
function get_cp_fixed(period, min)
    if period == "half_1"
        min <= 15 ? "H1_15" : (min <= 30 ? "H1_30" : "H1_45")
    elseif period == "half_2"
        min <= 15 ? "H2_15" : (min <= 30 ? "H2_30" : "H2_45")
    elseif period == "extra_time_1"
        "ET1_15"
    else
        "H2_45" # Fallback
    end
end

for d in [df_pass, df_press, df_shot]; d.checkpoint = get_cp_fixed.(d.period, d.minute); end

# 3. FEATURE ENGINEERING
# Pressure Resistance: Accuracy of passes under pressure
press_res = combine(groupby(df_press, [:player_appearance_id, :checkpoint]), :accurate => (x -> sum(x) / (length(x) + 1)) => :press_resistance)

# Verticality
pass_qual = combine(groupby(df_pass, [:player_appearance_id, :checkpoint]), :stage => (x -> sum(x .== "top") / (length(x) + 1)) => :verticality)

# Danger Index
shot_qual = combine(groupby(df_shot, [:player_appearance_id, :checkpoint]), :technique => (x -> sum(x .∈ [["volley", "header"]]) / (length(x) + 1)) => :danger_index)

# Join
df = leftjoin(df_main, press_res, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, pass_qual, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, shot_qual, on=[:player_appearance_id, :checkpoint])

# Coalesce
for c in [:press_resistance, :verticality, :danger_index]; df[!, c] = coalesce.(df[!, c], 0.0); end

# Filter and Drop
df = filter(row -> row.position != "G", df)
dropmissing!(df, [:scored_after, :last15_hsr])

# PCA on Volume
vol_feats = ["last15_sprints", "last15_hsr", "last15_distance"]
X_vol = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, vol_feats]), dims=1), Matrix{Float64}(df[:, vol_feats]))
pcs = MultivariateStats.transform(fit(PCA, collect(X_vol'); maxoutdim=2), collect(X_vol'))'

# Final Data
data = DataFrame(pcs, [:PC1, :PC2])
data.target = Float64.(df.scored_after)
data.press_resistance = df.press_resistance
data.verticality = df.verticality
data.danger_index = df.danger_index
data.position = df.position
data.shots_ot = Float64.(df.cumul_shots_on_target)
data.rel_intensity = df.last15_sprints ./ ((df.cumul_sprints ./ ((df.checkpoint_min ./ 15) .+ 1)) .+ 1)

# 4. EVALUATION
Random.seed!(123)
pos_c = sum(data.target .== 1); total = nrow(data)
wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in data.target]

formula = @formula(target ~ PC1 + PC2 + press_resistance + verticality + danger_index + position + shots_ot + rel_intensity)
m = glm(formula, data, Binomial(), LogitLink(), wts=wts)

println("--- IMPROVED MODEL RESULTS ---")
println("AUC: ", round(au_roccurve(data.target, predict(m)), digits=4))
println(coeftable(m))

# Hausman with is_home
data.is_home = Float64.(df.is_home)
m_s1 = lm(@formula(PC1 ~ verticality + danger_index + position + is_home), data)
data.v_hat = residuals(m_s1)
m_iv = glm(@formula(target ~ PC1 + v_hat + press_resistance + verticality + danger_index + position + shots_ot + rel_intensity), data, Binomial(), LogitLink(), wts=wts)

println("\nHausman Test (v_hat p-value): ", round(coeftable(m_iv).cols[4][findfirst(x->x=="v_hat", coefnames(m_iv))], digits=5))

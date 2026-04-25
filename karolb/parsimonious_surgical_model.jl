using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics, Random, Distributions

println("="^60)
println("PARSIMONIOUS MODEL SELECTION: SURGICAL VARIABLE REDUCTION")
println("="^60)

# 1. LOAD DATA
df_main = CSV.read("for_participants/data/players_quarters_final.csv", DataFrame)
df_pass = CSV.read("for_participants/data/player_appearance_pass.csv", DataFrame)
df_press = CSV.read("for_participants/data/player_appearance_behaviour_under_pressure.csv", DataFrame)
df_shot = CSV.read("for_participants/data/player_appearance_shot_limited.csv", DataFrame)

function get_cp_fixed(period, min)
    if period == "half_1"
        min <= 15 ? "H1_15" : (min <= 30 ? "H1_30" : "H1_45")
    elseif period == "half_2"
        min <= 15 ? "H2_15" : (min <= 30 ? "H2_30" : "H2_45")
    elseif period == "extra_time_1"
        "ET1_15"
    else
        "H2_45"
    end
end

for d in [df_pass, df_press, df_shot]; d.checkpoint = get_cp_fixed.(d.period, d.minute); end

# FEATURE ENGINEERING
press_res = combine(groupby(df_press, [:player_appearance_id, :checkpoint]), :accurate => (x -> sum(x) / (length(x) + 1)) => :press_resistance)
pass_qual = combine(groupby(df_pass, [:player_appearance_id, :checkpoint]), :stage => (x -> sum(x .== "top") / (length(x) + 1)) => :verticality)
shot_qual = combine(groupby(df_shot, [:player_appearance_id, :checkpoint]), :technique => (x -> sum(x .∈ [["volley", "header"]]) / (length(x) + 1)) => :danger_index)

df = leftjoin(df_main, press_res, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, pass_qual, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, shot_qual, on=[:player_appearance_id, :checkpoint])

for c in [:press_resistance, :verticality, :danger_index]; df[!, c] = coalesce.(df[!, c], 0.0); end
df = filter(row -> row.position != "G", df)
dropmissing!(df, [:scored_after, :last15_hsr, :last15_mean_max_speed])

# 2. REDUCED PCA (Keep only top 2 PCs for simplicity)
all_phys_vars = [
    "last15_sprints", "last15_hsr", "last15_distance", "last15_mean_max_speed", "last15_peak_speed",
    "cumul_sprints", "cumul_hsr", "cumul_distance", "cumul_mean_max_speed", "cumul_peak_speed"
]

X_all = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, all_phys_vars]), dims=1), Matrix{Float64}(df[:, all_phys_vars]))
pca_fit = fit(PCA, collect(X_all'); maxoutdim=2)
pcs = MultivariateStats.transform(pca_fit, collect(X_all'))'

data = DataFrame(pcs, [:PC1, :PC2])
data.target = Float64.(df.scored_after)
data.verticality = df.verticality
data.danger_index = df.danger_index
data.position = df.position

# 3. SURGICAL ESTIMATION (Strict significance check)
pos_c = sum(data.target .== 1); total = nrow(data)
wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in data.target]

println("\nFinal Parsimonious Model Estimation...")
m = glm(@formula(target ~ PC1 + PC2 + verticality + danger_index + position), data, Binomial(), LogitLink(), wts=wts)

# Print Detailed Results with P-Values
ct = coeftable(m)
println("\n[REGRESSION RESULTS]")
println(rpad("Variable", 15), " | ", rpad("Coef", 8), " | ", rpad("p-value", 8), " | ", "Signif")
println("-"^50)
for i in 1:size(ct.cols[1], 1)
    p = ct.cols[4][i]
    signif = p < 0.01 ? "***" : (p < 0.05 ? "**" : (p < 0.1 ? "*" : ""))
    println(rpad(ct.rownms[i], 15), " | ", rpad(round(ct.cols[1][i], digits=4), 8), " | ", rpad(round(p, digits=4), 8), " | ", signif)
end

# 4. FINAL BOOTSTRAP FOR ROBUST AUC
function run_parsimonious_bootstrap(df_in, n_boot=100)
    aucs = []
    for i in 1:n_boot
        df_b = df_in[sample(1:nrow(df_in), nrow(df_in), replace=true), :]
        pos_c = sum(df_b.target .== 1); total = nrow(df_b)
        wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in df_b.target]
        try
            m_b = glm(@formula(target ~ PC1 + PC2 + verticality + danger_index + position), df_b, Binomial(), LogitLink(), wts=wts)
            push!(aucs, au_roccurve(df_b.target, predict(m_b)))
        catch; end
    end
    println("\nParsimonious Mean AUC: ", round(mean(aucs), digits=4))
end

run_parsimonious_bootstrap(data)

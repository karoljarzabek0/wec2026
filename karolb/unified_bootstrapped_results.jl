using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics, Random, Distributions

println("="^60)
println("FINAL SYNCHRONIZED ANALYSIS: 2-FACTOR PARSIMONIOUS MODEL")
println("="^60)

# 1. LOAD & PREP
df_main = CSV.read("for_participants/data/players_quarters_final.csv", DataFrame)
df_pass = CSV.read("for_participants/data/player_appearance_pass.csv", DataFrame)
df_press = CSV.read("for_participants/data/player_appearance_behaviour_under_pressure.csv", DataFrame)
df_shot = CSV.read("for_participants/data/player_appearance_shot_limited.csv", DataFrame)

get_cp(p, m) = p == "half_1" ? (m <= 15 ? "H1_15" : (m <= 30 ? "H1_30" : "H1_45")) : (m <= 15 ? "H2_15" : (m <= 30 ? "H2_30" : "H2_45"))
for d in [df_pass, df_press, df_shot]; d.checkpoint = get_cp.(d.period, d.minute); end

pass_q = combine(groupby(df_pass, [:player_appearance_id, :checkpoint]), :stage => (x -> sum(x .== "top") / (length(x) + 1)) => :verticality)
shot_q = combine(groupby(df_shot, [:player_appearance_id, :checkpoint]), :technique => (x -> sum(x .∈ [["volley", "header"]]) / (length(x) + 1)) => :danger_index)

df = leftjoin(df_main, pass_q, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, shot_q, on=[:player_appearance_id, :checkpoint])
for c in [:verticality, :danger_index]; df[!, c] = coalesce.(df[!, c], 0.0); end
df = filter(row -> row.position != "G", df)
dropmissing!(df, [:scored_after, :last15_hsr, :last15_mean_max_speed, :cumul_hsr])

# 2. OPTIMAL 2-FACTOR PCA
phys_vars = ["last15_sprints", "last15_hsr", "last15_distance", "last15_mean_max_speed", "last15_peak_speed", "cumul_sprints", "cumul_hsr", "cumul_distance", "cumul_mean_max_speed", "cumul_peak_speed"]
X = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, phys_vars]), dims=1), Matrix{Float64}(df[:, phys_vars]))
pca = fit(PCA, collect(X'); maxoutdim=2)
pcs = MultivariateStats.transform(pca, collect(X'))'

data = DataFrame(pcs, [:PC1, :PC2])
data.target = Float64.(df.scored_after)
data.verticality = df.verticality
data.danger_index = df.danger_index
data.position = df.position
data.is_home = Float64.(df.is_home)

# 3. UNIFIED BOOTSTRAP (100 Iterations)
function run_final_boot(df_in, formula, name, is_iv=false)
    aucs = []
    coefs = []
    for i in 1:100
        df_b = df_in[sample(1:nrow(df_in), nrow(df_in), replace=true), :]
        if is_iv
            m_s1 = lm(@formula(PC1 ~ verticality + danger_index + position + is_home), df_b)
            df_b.v_hat = residuals(m_s1)
        end
        pos_c = sum(df_b.target .== 1); total = nrow(df_b)
        wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in df_b.target]
        try
            m = glm(formula, df_b, Binomial(), LogitLink(), wts=wts)
            push!(aucs, au_roccurve(df_b.target, predict(m)))
            push!(coefs, coef(m))
        catch; end
    end
    println("\nRESULTS: $name")
    println("Mean AUC: ", round(mean(aucs), digits=4))
    println("Coefficients (Sample): ", round.(mean(hcat(coefs...)', dims=1), digits=4))
end

run_final_boot(data, @formula(target ~ PC1 + PC2 + position), "Model C (Physical 2-Factor)")
run_final_boot(data, @formula(target ~ PC1 + PC2 + verticality + danger_index + position), "Model A (Hybrid 2-Factor)")
run_final_boot(data, @formula(target ~ PC1 + v_hat + PC2 + verticality + danger_index + position), "Final Model (Causal IV 2-Factor)", true)

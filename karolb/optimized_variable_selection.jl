using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics, Random, Distributions

println("="^60)
println("OPTIMIZING VARIABLE SELECTION: ALL-PHYSICAL PCA & AIC PRUNING")
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
dropmissing!(df, [:scored_after, :last15_hsr, :last15_mean_max_speed, :cumul_hsr])

# 2. ALL-PHYSICAL PCA (Including Cumulative)
all_phys_vars = [
    "last15_sprints", "last15_hsr", "last15_distance", "last15_mean_max_speed", "last15_peak_speed",
    "cumul_sprints", "cumul_hsr", "cumul_distance", "cumul_mean_max_speed", "cumul_peak_speed"
]

println("\nRunning PCA on all 10 physical performance metrics...")
X_all = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, all_phys_vars]), dims=1), Matrix{Float64}(df[:, all_phys_vars]))
pca_all = fit(PCA, collect(X_all'); maxoutdim=10)

# Selection Rule: Keep PCs explaining 90% variance
cum_var = cumsum(principalvars(pca_all)) ./ tvar(pca_all)
n_pcs = findfirst(v -> v >= 0.90, cum_var)
println("Selected $n_pcs PCs to explain $(round(cum_var[n_pcs]*100, digits=1))% of variance.")

pcs_final = MultivariateStats.transform(pca_all, collect(X_all'))'[ :, 1:n_pcs]
pc_names = [Symbol("PC$i") for i in 1:n_pcs]
data = DataFrame(pcs_final, pc_names)

# Targets & Context
data.target = Float64.(df.scored_after)
data.press_resistance = df.press_resistance
data.verticality = df.verticality
data.danger_index = df.danger_index
data.position = df.position
data.shots_ot = Float64.(df.cumul_shots_on_target)
data.rel_intensity = df.last15_sprints ./ ((df.cumul_sprints ./ ((df.checkpoint_min ./ 15) .+ 1)) .+ 1)
data.is_home = Float64.(df.is_home)

# 3. VARIABLE SELECTION (Model Pruning)
# We test a full model and check for significance
println("\n[VARIABLE PRUNING ANALYSIS]")
pos_c = sum(data.target .== 1); total = nrow(data)
wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in data.target]

full_formula = Term(:target) ~ sum(term.(pc_names)) + term(:verticality) + term(:danger_index) + term(:press_resistance) + term(:position) + term(:shots_ot) + term(:rel_intensity)
m_full = glm(full_formula, data, Binomial(), LogitLink(), wts=wts)

# Identify non-significant variables (p > 0.2)
p_vals = coeftable(m_full).cols[4]
varnames = coefnames(m_full)
to_keep = [varnames[1]] # keep intercept
for i in 2:length(varnames)
    if p_vals[i] < 0.2
        push!(to_keep, varnames[i])
    else
        println("Pruning: $(varnames[i]) (p=$(round(p_vals[i], digits=3)))")
    end
end

# 4. FINAL BOOTSTRAP WITH OPTIMIZED SET
function run_bootstrapped_optimized(df_in, pc_cols, n_boot=100)
    coefs = []
    aucs = []
    
    # We always keep position and PC1 for theoretical reasons
    formula = @formula(target ~ PC1 + PC2 + verticality + danger_index + rel_intensity + position)
    
    for i in 1:n_boot
        df_b = df_in[sample(1:nrow(df_in), nrow(df_in), replace=true), :]
        pos_c = sum(df_b.target .== 1); total = nrow(df_b)
        wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in df_b.target]
        try
            m = glm(formula, df_b, Binomial(), LogitLink(), wts=wts)
            push!(coefs, coef(m))
            push!(aucs, au_roccurve(df_b.target, predict(m)))
        catch; end
    end
    
    c_mat = hcat(coefs...)'
    means = mean(c_mat, dims=1)
    stds = std(c_mat, dims=1)
    
    println("\n" * "="^40)
    println("FINAL OPTIMIZED MODEL RESULTS")
    println("="^40)
    println("Mean AUC: ", round(mean(aucs), digits=4))
    println("-"^40)
    names = ["Intercept", "PC1", "PC2", "Verticality", "DangerIdx", "RelIntensity", "Pos:D", "Pos:M"]
    for i in 1:length(names)
        println(rpad(names[i], 15), " | ", round(means[i], digits=4), " (se: ", round(stds[i], digits=4), ")")
    end
end

run_bootstrapped_optimized(data, pc_names)

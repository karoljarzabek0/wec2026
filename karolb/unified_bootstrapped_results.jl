using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics, Random, Distributions

println("="^60)
println("UNIFIED BOOTSTRAPPED ANALYSIS: ENHANCED PCA & REGRESSION")
println("="^60)

# --- 1. DATA LOADING & ENHANCED PCA ---
println("\n[ENHANCED PCA DOCUMENTATION]")
println("Source Dataset: for_participants/data/players_quarters_final.csv")
# Expanded variable list to include speed metrics as requested
vol_feats = ["last15_sprints", "last15_hsr", "last15_distance", "last15_mean_max_speed", "last15_peak_speed"]
println("Variables Used: ", join(vol_feats, ", "))
println("Rationale: Capture both physical volume (distance/sprints) and movement quality (peak/mean speed).")

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

# PCA EXECUTION (Enhanced)
X_vol_mat = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, vol_feats]), dims=1), Matrix{Float64}(df[:, vol_feats]))
pca_fit = fit(PCA, collect(X_vol_mat'); maxoutdim=3)
pcs = MultivariateStats.transform(pca_fit, collect(X_vol_mat'))'

println("\nEnhanced PCA Component Loadings:")
loadings_df = DataFrame(Variable=vol_feats, PC1=projection(pca_fit)[:,1], PC2=projection(pca_fit)[:,2], PC3=projection(pca_fit)[:,3])
println(loadings_df)
println("Explained Variance: ", round(sum(principalvars(pca_fit))/tvar(pca_fit)*100, digits=2), "%")

data = DataFrame(pcs, [:PC1, :PC2, :PC3])
data.target = Float64.(df.scored_after)
data.press_resistance = df.press_resistance
data.verticality = df.verticality
data.danger_index = df.danger_index
data.position = df.position
data.shots_ot = Float64.(df.cumul_shots_on_target)
data.rel_intensity = df.last15_sprints ./ ((df.cumul_sprints ./ ((df.checkpoint_min ./ 15) .+ 1)) .+ 1)
data.is_home = Float64.(df.is_home)

# --- 2. UNIFIED BOOTSTRAP FUNCTION ---
function print_bootstrapped_results(name, coef_means, coef_stds, auc, names)
    println("\n" * "="^40)
    println("RESULTS: $name")
    println("="^40)
    println("Mean AUC: ", round(auc, digits=4))
    println("-"^40)
    println(rpad("Variable", 18), " | ", rpad("Coef", 8), " | ", rpad("Std.Err", 8), " | ", "z-stat")
    println("-"^40)
    for i in 1:length(names)
        z = coef_means[i] / coef_stds[i]
        println(rpad(names[i], 18), " | ", rpad(round(coef_means[i], digits=4), 8), " | ", rpad(round(coef_stds[i], digits=4), 8), " | ", round(z, digits=2))
    end
end

function run_bootstrapped_model(df_in, formula, name, n_boot=100)
    coefs = []
    aucs = []
    var_names = []
    
    for i in 1:n_boot
        df_b = df_in[sample(1:nrow(df_in), nrow(df_in), replace=true), :]
        pos_c = sum(df_b.target .== 1); total = nrow(df_b)
        wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in df_b.target]
        
        try
            m = glm(formula, df_b, Binomial(), LogitLink(), wts=wts)
            push!(coefs, coef(m))
            push!(aucs, au_roccurve(df_b.target, predict(m)))
            if i == 1; var_names = coefnames(m); end
        catch; end
    end
    
    c_mat = hcat(coefs...)'
    print_bootstrapped_results(name, mean(c_mat, dims=1), std(c_mat, dims=1), mean(aucs), var_names)
end

# --- 3. EXECUTION ---

# Model C
run_bootstrapped_model(data, @formula(target ~ PC1 + PC2 + PC3 + position + shots_ot), "Model C (Physical)")

# Model A
run_bootstrapped_model(data, @formula(target ~ PC1 + PC2 + PC3 + verticality + danger_index + position + shots_ot), "Model A (Hybrid)")

# Final IV
function run_bootstrapped_iv(df_in, n_boot=50)
    coefs = []
    aucs = []
    var_names = []
    for i in 1:n_boot
        df_b = df_in[sample(1:nrow(df_in), nrow(df_in), replace=true), :]
        m_s1 = lm(@formula(PC1 ~ verticality + danger_index + position + is_home), df_b)
        df_b.v_hat = residuals(m_s1)
        pos_c = sum(df_b.target .== 1); total = nrow(df_b)
        wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in df_b.target]
        try
            m = glm(@formula(target ~ PC1 + v_hat + press_resistance + verticality + danger_index + position + shots_ot + rel_intensity), df_b, Binomial(), LogitLink(), wts=wts)
            push!(coefs, coef(m))
            push!(aucs, au_roccurve(df_b.target, predict(m)))
            if i == 1; var_names = coefnames(m); end
        catch; end
    end
    c_mat = hcat(coefs...)'
    print_bootstrapped_results("Final Model (Causal IV)", mean(c_mat, dims=1), std(c_mat, dims=1), mean(aucs), var_names)
end

run_bootstrapped_iv(data)

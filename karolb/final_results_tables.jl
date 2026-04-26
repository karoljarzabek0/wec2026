# ==============================================================================
# FINAL RESULTS TABLES: PCA AND BOOTSTRAPPED LOGIT MODELS
# ==============================================================================

using Pkg

# 0. ROBUST ENVIRONMENT AND DEPENDENCY CHECK
# Use the directory of the script to find the project root
# Using variables instead of const to allow re-running in the same REPL session
SCRIPT_DIR = @__DIR__
PROJECT_ROOT = dirname(SCRIPT_DIR)

println("Activating environment at: $PROJECT_ROOT")
Pkg.activate(PROJECT_ROOT)

# Check for required packages and prompt installation if missing
required_packages = ["CSV", "DataFrames", "GLM", "StatsBase", "MultivariateStats", "EvalMetrics", "Statistics", "Random", "Distributions", "Plots"]
installed_packages = keys(Pkg.project().dependencies)
missing_packages = setdiff(required_packages, installed_packages)

if !isempty(missing_packages)
    println("\n" * "!"^60)
    println("MISSING PACKAGES DETECTED: ", join(missing_packages, ", "))
    println("Please run the following command in your Julia REPL to install them:")
    println("using Pkg; Pkg.add([\"" * join(missing_packages, "\", \"") * "\"])")
    println("!"^60 * "\n")
    
    # Critical check for basic operations
    if "CSV" in missing_packages || "DataFrames" in missing_packages
        error("Critical packages (CSV/DataFrames) missing. Please install them as shown above.")
    end
end

using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics, Random, Distributions, Plots
using Plots.Measures

# 1. DATA SETUP & PREPROCESSING
# Robust paths using absolute locations
DATA_DIR = joinpath(PROJECT_ROOT, "for_participants", "data")

function load_data()
    files = ["players_quarters_final.csv", "player_appearance_pass.csv", 
             "player_appearance_shot_limited.csv", "player_appearance_behaviour_under_pressure.csv"]
    
    for f in files
        full_path = joinpath(DATA_DIR, f)
        if !isfile(full_path)
            error("Data file not found: $full_path\nPlease ensure the 'for_participants/data' folder exists in the project root.")
        end
    end

    df_main = CSV.read(joinpath(DATA_DIR, "players_quarters_final.csv"), DataFrame)
    df_pass = CSV.read(joinpath(DATA_DIR, "player_appearance_pass.csv"), DataFrame)
    df_shot = CSV.read(joinpath(DATA_DIR, "player_appearance_shot_limited.csv"), DataFrame)
    df_press = CSV.read(joinpath(DATA_DIR, "player_appearance_behaviour_under_pressure.csv"), DataFrame)
    return df_main, df_pass, df_shot, df_press
end

df_main, df_pass, df_shot, df_press = load_data()

# Define time checkpoints (H1_15, H1_30, etc.)
get_cp(p, m) = p == "half_1" ? (m <= 15 ? "H1_15" : (m <= 30 ? "H1_30" : "H1_45")) : (m <= 15 ? "H2_15" : (m <= 30 ? "H2_30" : "H2_45"))

for d in [df_pass, df_shot, df_press]
    d.checkpoint = get_cp.(d.period, d.minute)
end

# Feature Engineering: Aggregate appearance data
pass_q = combine(groupby(df_pass, [:player_appearance_id, :checkpoint]), :stage => (x -> sum(x .== "top") / (length(x) + 1)) => :verticality)
shot_q = combine(groupby(df_shot, [:player_appearance_id, :checkpoint]), :technique => (x -> sum(x .∈ [["volley", "header"]]) / (length(x) + 1)) => :danger_index)
press_q = combine(groupby(df_press, [:player_appearance_id, :checkpoint]), :accurate => (x -> sum(x .== true) / (length(x) + 1)) => :press_resistance)

# Merge back to main dataframe
df = leftjoin(df_main, pass_q, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, shot_q, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, press_q, on=[:player_appearance_id, :checkpoint])

# Fill missing values and filter positions
for c in [:verticality, :danger_index, :press_resistance]
    df[!, c] = coalesce.(df[!, c], 0.0)
end
df = filter(row -> row.position != "G", df) # Exclude goalkeepers
dropmissing!(df, [:scored_after, :last15_hsr, :last15_mean_max_speed, :cumul_hsr])

# 2. PHYSICAL PCA (Dimensionality Reduction)
phys_vars = ["last15_sprints", "last15_hsr", "last15_distance", "last15_mean_max_speed", "last15_peak_speed", 
             "cumul_sprints", "cumul_hsr", "cumul_distance", "cumul_mean_max_speed", "cumul_peak_speed"]

X = StatsBase.transform(fit(ZScoreTransform, Matrix{Float64}(df[:, phys_vars]), dims=1), Matrix{Float64}(df[:, phys_vars]))
pca = fit(PCA, collect(X'); maxoutdim=2)
pcs = MultivariateStats.transform(pca, collect(X'))'

# Prepare data for modeling
data = DataFrame(pcs, [:PC1, :PC2])
data.target = Float64.(df.scored_after)
data.verticality = df.verticality
data.danger_index = df.danger_index
data.press_resistance = df.press_resistance
data.position = df.position
data.is_home = Float64.(df.is_home)
data.minute_in = Float64.(df.minute_in)
data.is_star = [v <= 11 ? 1.0 : 0.0 for v in df.jersey_number]
data.is_4231 = [v == "4-2-3-1" ? 1.0 : 0.0 for v in df.formation]
data.is_433 = [v == "4-3-3" ? 1.0 : 0.0 for v in df.formation]
data.is_back5 = [startswith(string(v), "5") ? 1.0 : 0.0 for v in df.formation]

# 3. UTILITY FUNCTIONS
function compute_metrics(target, probs)
    preds = probs .>= 0.5
    tp = sum((preds .== 1) .& (target .== 1))
    tn = sum((preds .== 0) .& (target .== 0))
    fp = sum((preds .== 1) .& (target .== 0))
    fn = sum((preds .== 0) .& (target .== 1))
    return (recall=tp/(tp+fn), precision=tp/(tp+fp))
end

function run_report(df_in, formula, name, is_iv=false, type=:structural)
    n_boot = 100
    all_coefs = []
    all_aucs = []
    n = nrow(df_in)
    
    if is_iv
        println("\n" * "="^60)
        println("FIRST-STAGE: minute_in -> is_star ($name)")
        println("="^60)
        if type == :full_tactic
            m_s1 = lm(@formula(is_star ~ PC1 + PC2 + position + is_home + is_4231 + is_433 + is_back5 + press_resistance + verticality + danger_index + minute_in), df_in)
            m_res = lm(@formula(is_star ~ PC1 + PC2 + position + is_home + is_4231 + is_433 + is_back5 + press_resistance + verticality + danger_index), df_in)
        elseif type == :hybrid
            m_s1 = lm(@formula(is_star ~ PC1 + PC2 + position + verticality + danger_index + minute_in), df_in)
            m_res = lm(@formula(is_star ~ PC1 + PC2 + position + verticality + danger_index), df_in)
        else
            m_s1 = lm(@formula(is_star ~ PC1 + PC2 + position + minute_in), df_in)
            m_res = lm(@formula(is_star ~ PC1 + PC2 + position), df_in)
        end
        df_in.v_hat_star = residuals(m_s1)
        f_stat = ((sum(residuals(m_res).^2) - sum(residuals(m_s1).^2))/1) / (sum(residuals(m_s1).^2)/(n-length(coef(m_s1))))
        println("Instrument F-Stat: ", round(f_stat, digits=2))
        println("-"^60)
    end
    
    for i in 1:n_boot
        df_b = df_in[sample(1:nrow(df_in), nrow(df_in), replace=true), :]
        if is_iv
            if type == :full_tactic
                df_b.v_hat_star = residuals(lm(@formula(is_star ~ PC1 + PC2 + position + is_home + is_4231 + is_433 + is_back5 + press_resistance + verticality + danger_index + minute_in), df_b))
            elseif type == :hybrid
                df_b.v_hat_star = residuals(lm(@formula(is_star ~ PC1 + PC2 + position + verticality + danger_index + minute_in), df_b))
            else
                df_b.v_hat_star = residuals(lm(@formula(is_star ~ PC1 + PC2 + position + minute_in), df_b))
            end
        end
        wts = [v == 1.0 ? nrow(df_b)/(2*sum(df_b.target .== 1)) : nrow(df_b)/(2*(nrow(df_b)-sum(df_b.target .== 1))) for v in df_b.target]
        try
            m = glm(formula, df_b, Binomial(), LogitLink(), wts=wts)
            push!(all_coefs, coef(m)); push!(all_aucs, au_roccurve(df_b.target, predict(m)))
        catch; end
    end
    
    means = mean(hcat(all_coefs...)', dims=1); stds = std(hcat(all_coefs...)', dims=1)
    m_full = glm(formula, df_in, Binomial(), LogitLink(), wts=[v == 1.0 ? n/(2*sum(df_in.target .== 1)) : n/(2*(n-sum(df_in.target .== 1))) for v in df_in.target])
    metrics = compute_metrics(df_in.target, predict(m_full))
    
    println("\nBOOTSTRAPPED RESULTS: $name")
    res = DataFrame(Variable = coefnames(m_full), Estimate = round.(vec(means), digits=4), StdErr = round.(vec(stds), digits=4), p_val = [round(2 * (1 - cdf(Normal(), abs(z))), digits=4) for z in vec(means ./ stds)])
    res.Sig = [p < 0.01 ? "***" : (p < 0.05 ? "**" : (p < 0.1 ? "*" : "")) for p in res.p_val]
    println(res)
    
    println("\n[DIAGNOSTICS SUMMARY]")
    mean_auc = mean(all_aucs)
    println("Mean Bootstrapped AUC:         ", round(mean_auc, digits=4))
    if is_iv 
        dwh_p = res.p_val[findfirst(x->x=="v_hat_star", res.Variable)]
        println("Durbin-Wu-Hausman (DWH) p-val: ", round(dwh_p, digits=4))
        println("DWH Verdict:                   ", dwh_p < 0.05 ? "Endogeneity Detected (IV Required)" : "Exogeneity Not Rejected")
    end
    return (name=name, auc=mean_auc, recall=metrics.recall, precision=metrics.precision)
end

# 4. EXECUTION (GENERATING MODELS FOR REPORT)
println("\n" * "="^80)
println("FINAL ALIGNMENT REPORT (MODELS REPLICATING PDF TABLES)")
println("="^80)

summary = []

# Table 4 models
push!(summary, run_report(data, @formula(target ~ PC1 + PC2 + position), "Model A: Baseline (No Proxy)", false, :structural))
push!(summary, run_report(data, @formula(target ~ PC1 + PC2 + verticality + danger_index + position), "Model B: Hybrid (No Proxy)", false, :hybrid))

# Table 5 models
push!(summary, run_report(data, @formula(target ~ PC1 + PC2 + position + is_star), "Model C: Baseline (With Star Proxy)", false, :structural))
push!(summary, run_report(data, @formula(target ~ PC1 + PC2 + verticality + danger_index + position + is_star), "Model D: Hybrid (With Star Proxy)", false, :hybrid))

# Table 6 models
push!(summary, run_report(data, @formula(target ~ PC1 + PC2 + position + v_hat_star + is_star), "Model E: IV Logit Structural", true, :structural))

# Table 7 models
push!(summary, run_report(data, @formula(target ~ PC1 + PC2 + position + is_home + is_4231 + is_433 + is_back5 + press_resistance + verticality + danger_index + v_hat_star + is_star), "Model F: Tactical IV Champion", true, :full_tactic))

println("\n" * "="^80)
println("TABLE 8: FULL PERFORMANCE SUMMARY")
println("="^80)
final_df = DataFrame(Model = [r.name for r in summary], AUC = [round(r.auc, digits=4) for r in summary], Recall_Pct = [round(r.recall * 100, digits=2) for r in summary], Precision_Pct = [round(r.precision * 100, digits=2) for r in summary])
println(final_df)
println("\n" * "="^80)
println("END OF REPORT")
println("="^80)

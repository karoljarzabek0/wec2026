using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, StatsBase, MultivariateStats, EvalMetrics, Statistics, Random, Distributions

# 1. DATA SETUP
df_main = CSV.read("for_participants/data/players_quarters_final.csv", DataFrame)
df_pass = CSV.read("for_participants/data/player_appearance_pass.csv", DataFrame)
df_shot = CSV.read("for_participants/data/player_appearance_shot_limited.csv", DataFrame)

get_cp(p, m) = p == "half_1" ? (m <= 15 ? "H1_15" : (m <= 30 ? "H1_30" : "H1_45")) : (m <= 15 ? "H2_15" : (m <= 30 ? "H2_30" : "H2_45"))
for d in [df_pass, df_shot]; d.checkpoint = get_cp.(d.period, d.minute); end

pass_q = combine(groupby(df_pass, [:player_appearance_id, :checkpoint]), :stage => (x -> sum(x .== "top") / (length(x) + 1)) => :verticality)
shot_q = combine(groupby(df_shot, [:player_appearance_id, :checkpoint]), :technique => (x -> sum(x .∈ [["volley", "header"]]) / (length(x) + 1)) => :danger_index)

df = leftjoin(df_main, pass_q, on=[:player_appearance_id, :checkpoint])
df = leftjoin(df, shot_q, on=[:player_appearance_id, :checkpoint])
for c in [:verticality, :danger_index]; df[!, c] = coalesce.(df[!, c], 0.0); end
df = filter(row -> row.position != "G", df)
dropmissing!(df, [:scored_after, :last15_hsr, :last15_mean_max_speed, :cumul_hsr])

# 2. PCA DECOMPOSITION
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
data.checkpoint_min = Float64.(df.checkpoint_min)
data.minute_in = Float64.(df.minute_in)
data.subbed = df.subbed
data.jersey_number = df.jersey_number

# 3. ADVANCED DIAGNOSTICS & ROBUST FORMATTING
# Create 'time_on_pitch' (Fatigue proxy) and ensure 'subbed' is numeric
data.time_on_pitch = data.checkpoint_min .- data.minute_in
data.is_sub = [v == true ? 1.0 : 0.0 for v in data.subbed]

function test_normality(residuals)
    # Jarque-Bera Test (Manual)
    n = length(residuals)
    m1 = mean(residuals)
    m2 = sum((residuals .- m1).^2) / n
    m3 = sum((residuals .- m1).^3) / n
    m4 = sum((residuals .- m1).^4) / n
    skew = m3 / (m2^1.5)
    kurt = m4 / (m2^2)
    jb_stat = (n/6) * (skew^2 + 0.25*(kurt-3)^2)
    p_val = 1 - cdf(Chisq(2), jb_stat)
    return round(p_val, digits=5)
end

function test_hetero(m, target)
    y = target; mu = predict(m)
    r_pearson = (y .- mu) ./ sqrt.(mu .* (1.0 .- mu) .+ 1e-6)
    r2 = r_pearson .^ 2
    test_mod = lm(hcat(ones(length(mu)), mu), r2)
    return round(coeftable(test_mod).cols[4][2], digits=5)
end

function compute_classification_metrics(target, probs)
    # Threshold at 0.5 (Standard for balanced models)
    preds = probs .>= 0.5
    tp = sum((preds .== 1) .& (target .== 1))
    tn = sum((preds .== 0) .& (target .== 0))
    fp = sum((preds .== 1) .& (target .== 0))
    fn = sum((preds .== 0) .& (target .== 1))
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    acc = (tp + tn) / length(target)
    
    return (acc=acc, recall=recall, precision=precision, tp=tp, tn=tn, fp=fp, fn=fn)
end

# Unified Bootstrapped Result Generator (Adjusts for Hetero and Non-Normality)
function run_robust_report(df_in, formula, name, is_iv=false)
    n_boot = 100
    all_coefs = []
    all_aucs = []
    f_stat = 0.0
    sargan_p = 1.0
    
    # 3.1 INSTRUMENT SELECTION
    df_in.time_on_pitch = Float64.(df_in.checkpoint_min .- df_in.minute_in)
    df_in.is_sub = [v == true ? 1.0 : 0.0 for v in df_in.subbed]
    df_in.jersey_raw = Float64.(df_in.jersey_number)
    
    if is_iv
        println("\n" * "="^60)
        println("FIRST-STAGE REGRESSION (OLS): MODELING EXOGENOUS EFFORT")
        println("="^60)
        m_s1_full = lm(@formula(PC1 ~ verticality + danger_index + position + is_sub + time_on_pitch + jersey_raw), df_in)
        ct1 = coeftable(m_s1_full)
        fs_table = DataFrame(Variable=ct1.rownms, Coef=round.(ct1.cols[1], digits=4), p_val=round.(ct1.cols[4], digits=4))
        fs_table.Sig = [p < 0.001 ? "***" : (p < 0.05 ? "**" : (p < 0.1 ? "*" : "")) for p in fs_table.p_val]
        println(fs_table)
        df_in.v_hat = residuals(m_s1_full)
        m_s1_restricted = lm(@formula(PC1 ~ verticality + danger_index + position), df_in)
        ssr_r = sum(residuals(m_s1_restricted).^2); ssr_u = sum(residuals(m_s1_full).^2)
        n = nrow(df_in); k = length(coef(m_s1_full)); q = 3 
        f_stat = ((ssr_r - ssr_u)/q) / (ssr_u/(n-k))
        pos_c = sum(df_in.target .== 1); total = nrow(df_in)
        wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in df_in.target]
        m_overid = glm(@formula(target ~ PC1 + PC2 + v_hat + verticality + danger_index + position + is_sub + time_on_pitch + jersey_raw), df_in, Binomial(), LogitLink(), wts=wts)
        m_base = glm(@formula(target ~ PC1 + PC2 + v_hat + verticality + danger_index + position), df_in, Binomial(), LogitLink(), wts=wts)
        lr_stat = deviance(m_base) - deviance(m_overid)
        sargan_p = 1 - cdf(Chisq(3), lr_stat)
        println("\nInstrument Strength (F-Stat): ", round(f_stat, digits=2))
        println("Sargan (Overid p-val):        ", round(sargan_p, digits=4))
        println("-"^60)
    end
    
    for i in 1:n_boot
        df_b = df_in[sample(1:nrow(df_in), nrow(df_in), replace=true), :]
        if is_iv
            m_s1 = lm(@formula(PC1 ~ verticality + danger_index + position + is_sub + time_on_pitch + jersey_raw), df_b)
            df_b.v_hat = residuals(m_s1)
        end
        pos_c = sum(df_b.target .== 1); total = nrow(df_b)
        wts = [v == 1.0 ? total/(2*pos_c) : total/(2*(total-pos_c)) for v in df_b.target]
        try
            m = glm(formula, df_b, Binomial(), LogitLink(), wts=wts)
            push!(all_coefs, coef(m))
            push!(all_aucs, au_roccurve(df_b.target, predict(m)))
        catch; end
    end
    
    means = mean(hcat(all_coefs...)', dims=1); stds = std(hcat(all_coefs...)', dims=1)
    pos_c_full = sum(df_in.target .== 1); total_full = nrow(df_in)
    wts_full = [v == 1.0 ? total_full/(2*pos_c_full) : total_full/(2*(total_full-pos_c_full)) for v in df_in.target]
    m_full = glm(formula, df_in, Binomial(), LogitLink(), wts=wts_full)
    probs = predict(m_full); metrics = compute_classification_metrics(df_in.target, probs)

    println("\n" * "-"^60); if is_iv println("ROBUST TWO-STAGE (IV) LOGIT RESULTS: $name") else println("ROBUST LOGIT ESTIMATES: $name") end; println("-"^60)
    res = DataFrame(Variable = coefnames(m_full), Estimate = round.(vec(means), digits=4), Robust_StdErr = round.(vec(stds), digits=4), z_stat = round.(vec(means ./ stds), digits=2))
    res.p_val = [round(2 * (1 - cdf(Normal(), abs(z))), digits=4) for z in res.z_stat]
    res.Sig = [p < 0.001 ? "***" : (p < 0.05 ? "**" : (p < 0.1 ? "*" : "")) for p in res.p_val]
    println(res)
    
    println("\n[DIAGNOSTICS SUMMARY]")
    mean_auc = mean(all_aucs)
    println("Mean Bootstrapped AUC:      ", round(mean_auc, digits=4))
    println("Heteroscedasticity p-val:   ", test_hetero(m_full, df_in.target))
    if is_iv println("Hausman Exogeneity p-val:   ", res.p_val[findfirst(x->x=="v_hat", res.Variable)]) end

    return (name=name, auc=mean_auc, recall=metrics.recall, precision=metrics.precision)
end

# 4. PRINT OFFICIAL REPORT
println("\n" * "="^80)
println("OFFICIAL ROBUST MODEL REPORT: WARSAW ECONOMETRIC CHALLENGE 2026")
println("="^80)

summary_results = []

# Execute Models and collect for summary
push!(summary_results, run_robust_report(data, @formula(target ~ PC1 + PC2 + position), "Model C (Baseline)"))
push!(summary_results, run_robust_report(data, @formula(target ~ PC1 + PC2 + verticality + danger_index + position), "Model A (Hybrid)"))
push!(summary_results, run_robust_report(data, @formula(target ~ PC1 + PC2 + v_hat + verticality + danger_index + position), "Final Model (Causal IV)", true))

println("\n" * "="^80)
println("FINAL CONSOLIDATED PERFORMANCE SUMMARY")
println("="^80)
summary_df = DataFrame(
    Model = [r.name for r in summary_results],
    Mean_AUC = [round(r.auc, digits=4) for r in summary_results],
    Recall = [round(r.recall * 100, digits=2) for r in summary_results],
    Precision = [round(r.precision * 100, digits=2) for r in summary_results]
)
println(summary_df)

println("\n" * "="^80)
println("END OF REPORT (All estimates adjusted for Hetero/Non-Normality via Bootstrapping)")
println("="^80)

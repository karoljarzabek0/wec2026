# ==============================================================================
# PHYSICAL FACTOR ANALYSIS: PCA AND CORRELATION
# ==============================================================================

using CSV, DataFrames, MultivariateStats, StatsBase, Plots, Plots.Measures, LinearAlgebra

# --- 1. CONFIGURATION ---
# Using relative paths for better portability
input_file = joinpath("..", "for_participants", "data", "players_quarters_final.csv")
export_dir = "output_physical"
mkpath(export_dir)

# Load data
if !isfile(input_file)
    error("Data file not found at: $input_file. Please ensure the script is run from the 'karolb' directory.")
end
df = CSV.read(input_file, DataFrame)

# --- 2. ANALYTICAL FUNCTION ---
function analyze_physical_factors(data_df, prefix, n_pcs)
    # A. Data Preparation and Standardization
    # Remove rows with missing values
    clean_df = dropmissing(data_df)
    matrix = Matrix{Float64}(clean_df)
    
    # Standardization (Z-Score) - essential for PCA
    dt = fit(ZScoreTransform, matrix, dims=1)
    std_data = StatsBase.transform(dt, matrix)
    
    # B. Input Variable Correlation Matrix
    cor_mat_vars = cor(std_data)
    f_names = names(clean_df)
    
    p_corr = heatmap(f_names, f_names, cor_mat_vars,
                    title="Variable Correlation Matrix ($prefix)",
                    colors=cgrad(:coolwarm), clim=(-1, 1),
                    xrotation=45, size=(900, 800),
                    left_margin=20mm, bottom_margin=20mm)

    # C. PCA Calculations
    # MultivariateStats expects data in format: variables x observations (collect(std_data'))
    pca_model = fit(PCA, collect(std_data'); maxoutdim=n_pcs)
    
    # Isolated factors (Scores)
    factors_scores = MultivariateStats.transform(pca_model, collect(std_data'))
    
    # D. Variance Statistics
    v_ratio = principalvars(pca_model) ./ tvar(pca_model)
    c_ratio = cumsum(v_ratio)
    
    # E. Loadings - Correlation of variables with factors
    pc_labels = ["PC$i" for i in 1:n_pcs]
    loadings_mat = cor(std_data, factors_scores', dims=1)
    
    # F. Plots
    # 1. Scree plot
    p_var = bar(1:n_pcs, v_ratio, title="Explained Variance ($prefix)", 
                xticks=1:n_pcs, label="Individual", color=:skyblue)
    plot!(c_ratio, label="Cumulative", marker=:circle, color=:orange)

    # 2. Loadings heatmap (key for interpreting factors)
    h_height = max(600, length(f_names) * 30)
    p_loadings = heatmap(pc_labels, f_names, loadings_mat,
                        title="PCA Loadings ($prefix)",
                        colors=cgrad(:RdYlGn, rev=true), clim=(-1, 1),
                        yflip=true, left_margin=30mm,
                        size=(400 + (n_pcs*100), h_height))

    # G. Result Tables Preparation
    loadings_df = DataFrame(loadings_mat, Symbol.(pc_labels))
    loadings_df[!, :Variable] = f_names
    select!(loadings_df, :Variable, All())

    variance_df = DataFrame(
        Component = pc_labels,
        Individual_Variance = v_ratio,
        Cumulative_Variance = c_ratio
    )

    # Transformation results (factors for predictive modeling)
    scores_df = DataFrame(factors_scores', Symbol.(pc_labels))

    return p_corr, p_var, p_loadings, loadings_df, variance_df, scores_df
end

# --- 3. PHYSICAL VARIABLE SELECTION ---
# Automatically exclude IDs, metadata, and shooting statistics
blacklist = ["player_id", "fixture_id", "player_appearance_id", "jersey_number", "is_home", "checkpoint_min", "scored_after"]
all_num = select(df, findall(col -> eltype(col) <: Number, eachcol(df)))

# Select columns that are not blacklisted and do not concern shots
phys_cols = [n for n in names(all_num) if !(n in blacklist) && !occursin("shot", lowercase(n))]
phys_df = select(all_num, phys_cols)

# --- 4. EXECUTION ---
num_factors = 3
p_corr, p_var, p_load, l_df, v_df, s_df = analyze_physical_factors(phys_df, "Physical", num_factors)

# --- 5. RESULTS EXPORT ---
savefig(p_corr, joinpath(export_dir, "physical_correlation_matrix.png"))
savefig(p_var, joinpath(export_dir, "physical_pca_variance.png"))
savefig(p_load, joinpath(export_dir, "physical_pca_loadings.png"))

CSV.write(joinpath(export_dir, "physical_loadings.csv"), l_df)
CSV.write(joinpath(export_dir, "physical_variance_summary.csv"), v_df)
CSV.write(joinpath(export_dir, "physical_factors_scores.csv"), s_df)

println("Analysis complete. Results saved in: $export_dir")

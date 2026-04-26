# ==============================================================================
# GLOBAL CORRELATION MATRIX: HEATMAP GENERATION
# ==============================================================================

using CSV, DataFrames, StatsBase, Plots, Plots.Measures

# 1. Configuration & Paths
# Automatically locate the project root relative to this script
SCRIPT_DIR = @__DIR__
PROJECT_ROOT = dirname(SCRIPT_DIR)

input_file = joinpath(PROJECT_ROOT, "for_participants", "data", "players_quarters_final.csv")
output_dir = joinpath(SCRIPT_DIR, "output")
mkpath(output_dir)

# 2. Load Data
if !isfile(input_file)
    error("Data file not found at: $input_file. Please ensure the 'for_participants/data' folder exists in the project root.")
end

df = CSV.read(input_file, DataFrame)

# 3. Select All Numeric Columns
all_numeric_df = select(df, findall(col -> eltype(col) <: Number, eachcol(df)))
dropmissing!(all_numeric_df)

feature_names = names(all_numeric_df)
total_vars = length(feature_names)

# 4. Calculate Correlation
global_cor_matrix = cor(Matrix{Float64}(all_numeric_df))

# 5. Visualization Settings
# Dynamic canvas size based on the number of variables
canvas_size = total_vars * 35 

p_global = heatmap(
    1:total_vars, 
    1:total_vars, 
    global_cor_matrix,
    xticks = (1:total_vars, feature_names),
    yticks = (1:total_vars, feature_names),
    title = "Global Correlation Matrix",
    
    # Using :viridis_r for a clear, high-contrast color scale
    colors = :viridis_r, 
    
    clim = (-1, 1),
    xrotation = 90,
    yflip = true,
    tickfontsize = 8,
    left_margin = 50mm,
    bottom_margin = 50mm,
    size = (canvas_size + 400, canvas_size + 100)
)

# 6. Save and Display
output_png = joinpath(output_dir, "global_correlation_reversed.png")
output_svg = joinpath(output_dir, "global_correlation_matrix.svg")

savefig(p_global, output_png)
savefig(p_global, output_svg)

println("Success! Heatmaps generated in: $output_dir")

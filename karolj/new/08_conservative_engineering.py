import pandas as pd
import numpy as np
import os

def engineer_conservative_features():
    print("Loading datasets...")
    data_dir = "for_participants/data"
    main_df = pd.read_csv(os.path.join(data_dir, "players_quarters_final.csv"))
    passes = pd.read_csv(os.path.join(data_dir, "player_appearance_pass.csv"))
    press = pd.read_csv(os.path.join(data_dir, "player_appearance_behaviour_under_pressure.csv"))
    
    # 1. Map Checkpoints
    period_map = {'half_1': 1, 'half_2': 2, 'extra_time_1': 3, 'extra_time_2': 4}
    passes['period_val'] = passes['period'].map(period_map)
    press['period_val'] = press['period'].map(period_map)
    main_df['period_val'] = main_df['checkpoint_period'].map(period_map)
    
    # Pre-group for performance
    pass_groups = {name: group for name, group in passes.groupby('player_appearance_id')}
    press_groups = {name: group for name, group in press.groupby('player_appearance_id')}
    
    def get_tactical_features(row):
        p_id = row['player_appearance_id']
        c_p = row['period_val']
        c_m = row['checkpoint_min']
        
        # Default results
        res = {
            'l15_top_third_pass_acc': 0.0,
            'l15_press_forward_rate': 0.0,
            'cumul_explosivity_ratio': 0.0
        }
        
        # Explosivity Ratio (from main_df directly - conservative)
        total_runs = row['cumul_sprints'] + row['cumul_hsr']
        if total_runs > 0:
            res['cumul_explosivity_ratio'] = row['cumul_sprints'] / total_runs
            
        # Passing: Top Third Accuracy
        if p_id in pass_groups:
            p_passes = pass_groups[p_id]
            l15_mask = (p_passes['period_val'] == c_p) & (p_passes['minute'] >= (c_m - 15)) & (p_passes['minute'] < c_m)
            l15_top = p_passes[l15_mask & (p_passes['stage'] == 'top')]
            if not l15_top.empty:
                res['l15_top_third_pass_acc'] = l15_top['accurate'].mean()
                
        # Pressure: Forward Pass Rate
        if p_id in press_groups:
            p_press = press_groups[p_id]
            l15_mask = (p_press['period_val'] == c_p) & (p_press['minute'] >= (c_m - 15)) & (p_press['minute'] < c_m)
            l15_p = p_press[l15_mask]
            if not l15_p.empty:
                res['l15_press_forward_rate'] = (l15_p['press_induced_outcome'] == 'forward_pass').mean()
                
        return pd.Series(res)

    print("Iterating to extract features...")
    tactical_feat = main_df.apply(get_tactical_features, axis=1)
    
    # Merge and Save
    final_df = pd.concat([main_df, tactical_feat], axis=1)
    final_df.to_csv("karolj/new/players_conservative_tactical.csv", index=False)
    print(f"Saved to karolj/new/players_conservative_tactical.csv. Shape: {final_df.shape}")

if __name__ == "__main__":
    engineer_conservative_features()

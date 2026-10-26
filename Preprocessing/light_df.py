# pick series with atleast either Fluid_Sensitive == True or Fat_Suppression == True 
import pandas as pd 

def pick_best_series(group):
    best = group[(group.Fluid_Sensitive == True) & (group.Fat_Suppression == True)]
    if len(best) == 0:
        best = group[group.Fluid_Sensitive == True]   # fallback: fluid-sensitive only
    if len(best) == 0:
        best = group                                  # fallback: take whatever exists
    return best.iloc[0]

def create_light_df(SERIES_MERGED):
    complete_series_merged = SERIES_MERGED.copy()
    LIGHT_DF = (
        complete_series_merged
        .groupby(['StudyInstanceUID', 'Anatomical_Plane'])
        .apply(pick_best_series)
        .reset_index(drop=True)
    )
    return LIGHT_DF
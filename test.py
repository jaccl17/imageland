# %% [markdown]
# # Title
#
# Some text
# %%
import numpy as np
# %%
apple_music = {'mag_bay': 1, 'ethel_cain': 2, 'billie_eilish': 3, 'ariana_grande': 4, 'clairo': 5, 
             'weyes_blood': 6, 'fka_twigs': 7, 'kelela': 8, 'charli_xcx': 9, 'halsey': 10, 'tinashe': 11, 
             'troye_sivan': 12, 'shygirl': 13, 'pinkpantheress': 14, 'emmylou_harris': 15}
spotify = {'ariana_grande': 1, 'beyonce': 2, 'hans_zimmer': 3, 'kelela': 4, 'ethel_cain': 5, 
             'pinkpantheress': 6, 'shygirl': 7, 'tyla': 8, 'billie_eilish': 9, 'one_direction': 10, 'aphex_twin': 11, 
             'lil_hero': 12, 'waxahatchee': 13, 'dua_lipa': 14, 'bee_gees': 15}
# %%
all_artists = set(apple_music.keys()).union(spotify.keys())
worst_rank1 = max(apple_music.values(), default=0) + 1
worst_rank2 = max(spotify.values(), default=0) + 1

averaged_rankings = {}
for artists in all_artists:
    rank1 = apple_music.get(artists, worst_rank1)  # Default to worst rank if missing
    rank2 = spotify.get(artists, worst_rank2)  # Default to worst rank if missing
    averaged_rankings[artists] = (rank1 + rank2) / 2

# Sort fruits by their averaged rankings
overall_ranking = sorted(averaged_rankings.items(), key=lambda x: x[1])
# %%
print("Overall Ranking:")
for rank, (fruit, avg_rank) in enumerate(overall_ranking, start=1):
    print(f"{rank}: {fruit} (Average Rank: {avg_rank:.2f})")
# %%
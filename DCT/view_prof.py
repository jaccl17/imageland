import pstats

with open("profile_output.txt", "w") as f:
    p = pstats.Stats('/home/unitx/wabbit_playground/DCT/profile_output.prof', stream=f)
    p.sort_stats('cumulative').print_stats() 
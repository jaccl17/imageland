def calculate_new_resolution(res_x, res_y, len_x, len_y, len_x_new, len_y_new):
    res_x_new = res_x * (len_x / len_x_new)
    res_y_new = res_y * (len_y / len_y_new)
    return int(res_x_new), int(res_y_new)

# Example test
print(calculate_new_resolution(res_x=2472, res_y=2064, len_x=248, len_y=211, len_x_new=500, len_y_new=425))
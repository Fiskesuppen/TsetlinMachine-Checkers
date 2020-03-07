white_vertical_multiplied = 0
white_vertical_summed = 0

for q in range(len(white_vertical_weights)):
    white_vertical_multiplied += white_vertical_weights[q] * (q + 1)
    white_vertical_summed += white_vertical_weights[q]
white_vertical_center = round_down(white_vertical_multiplied / white_vertical_summed)
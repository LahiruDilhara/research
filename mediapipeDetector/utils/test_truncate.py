def truncate(total_frames, window_size=5, step=3):
    if total_frames < window_size:
        return 0
    return ((total_frames - window_size) // step) * step + window_size

print(truncate(130))
print(truncate(5))
print(truncate(4))

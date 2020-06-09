encoded_lookup = Dict(value => idx for (idx, value) in enumerate(unique_sorted_labels))
labels = [encoded_lookup[label] for label in labels]

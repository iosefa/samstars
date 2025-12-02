import matplotlib.pyplot as plt

# Data
species = ["ōhi‘a", "silk oak", "koa", "lama", "kiawe", "other"]
counts  = [166772, 26819, 56490, 16114, 44343, 22695]
total = sum(counts)
[print((i/total) * 100) for i in counts]
print(total)

# Create the bar chart
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.bar(species, counts, width=0.6)
#
# # Labels and title (optional)
# ax.set_xlabel("Species")
# ax.set_ylabel("Count")
# ax.set_title("Tree Counts by Species")
#
# plt.tight_layout()
# plt.show()
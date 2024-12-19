import matplotlib.pyplot as plt

# Data
steps = [0, 50, 100, 150, 200, 250]
gradient_difference = [0.025805, 0.016587, 0.015965, 0.099982, 0.364184, 0.100790]

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(steps, gradient_difference, marker='o', linestyle='-', linewidth=2, color='b')
plt.title("Gradient Difference Over Steps")
plt.xlabel("Steps")
plt.ylabel("Gradient Difference")
plt.grid(True)

# Save the image
plt.savefig("gradient_difference_plot.png", dpi=300, bbox_inches="tight")
plt.show()

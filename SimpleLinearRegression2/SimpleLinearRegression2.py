import matplotlib.pyplot as plt
import numpy as np

# area
x = np.array([[73.75, 75.75, 76.5, 79.1, 81.5, 82.5, 84.85, 85.86, 87.5, 89.7, 90.91]]).T
# price
y = np.array([[1.49, 1.50, 1.51, 1.54, 1.58, 1.59, 1.63, 1.64, 1.61, 1.63, 1.64]]).T


def calculateB0B1(x, y):
    # Calculate the average
    xbar = np.mean(x)
    ybar = np.mean(y)
    x2bar = np.mean(x ** 2)
    x_ybar = np.mean(x * y)

    # Calculate b0, b1
    b1 = (xbar * ybar - x_ybar) / (xbar ** 2 - x2bar)
    b0 = ybar - b1 * xbar
    return b1, b0


# Calculate b0, b1
b1, b0 = calculateB0B1(x, y)
print("b1:", b1)
print("b0:", b0)

y_predicted = b0 + b1 * x
print(y_predicted)


# Visualize data
def showGraph(x, y, y_predicted, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(14, 8))
    plt.plot(x, y, 'r-o', label="price")
    plt.plot(x, y_predicted, 'b-x', label="predicted value")

    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)

    # Mean y value
    ybar = np.mean(y)

    plt.axhline(ybar, linestyle='--', linewidth=4, label="mean")
    plt.axis([x_min * 0.95, x_max * 1.05, y_min * 0.95, y_max * 1.05])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.text(x_min + 1.03, ybar, "mean", fontsize=16)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.show()


showGraph(x, y, y_predicted,
          title="House price by Area",
          xlabel="Area (m2)",
          ylabel="Price (Billion VND)")

import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(x, y):
    m = b = 0
    iteration = 100
    n = len(x)
    learning_rate = 0.001
    for i in range(iteration):
        y_pred = m * x + b
        cost = (1.0 / n) * sum([val**2 for val in (y - y_pred)])
        dm = (-2 / n) * sum(x * (y - y_pred))
        db = (-2 / n) * sum((y - y_pred))
        m = m - learning_rate * dm
        b = b - learning_rate * db
        plt.scatter(x, y, color="m",
                    marker="o", s=30)
        print (i, cost, m, b)
        plt.plot(x, y_pred, color="g")
        # putting labels
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


def main():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 8, 11, 14, 17])
    gradient_descent(x, y)


if __name__ == "__main__":
    main()

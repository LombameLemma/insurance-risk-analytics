import matplotlib.pyplot as plt

def plot_histogram(data, column):
    data[column].hist()
    plt.title(column)
    plt.show()
    
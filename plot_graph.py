import matplotlib.pyplot as plt


def plot(scores, mean_scores):
    plt.clf()
    plt.title("Training")
    plt.xlabel("Number of games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.draw()
    plt.show(block=False)
    plt.pause(0.1)
    plt.pause(0.1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TrainingStats:
    def __init__(self, house: str, loops: int):
        self.house = house
        self.loops = loops
        self.records = np.zeros(loops)

    def getLoss(self):
        return self.records[-1]

    def plot(self):
        df = pd.DataFrame(self.records.T, columns=["loss"])

        iterations = np.arange(len(df))

        fig, ax1 = plt.subplots()

        # Loss
        line = ax1.plot(iterations, df["loss"], color="blue", label="Loss")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")

        ax1.legend(line, ["Loss"])

        fig.suptitle(self.house + "Training Statistics")
        plt.savefig(self.house + "_loss.png")
        plt.close()

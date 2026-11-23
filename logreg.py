import numpy as np
import random
from loss_plot import TrainingStats
import ft_math as fm


class LogReg:
    def __init__(
        self,
        notes: np.ndarray,
        house: str,
        students_houses: np.ndarray,
    ):
        self.notes = notes
        self.house = house
        self.houses = students_houses
        self.weights = np.zeros(notes.shape[1])
        self.bias: float = 0.0
        self.learning_rate: float = 0.1

    def cost_function(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)

        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def gradient_descent(
        self,
        errors: np.ndarray,
        notes: np.ndarray,
    ):
        gradients = fm.mean(errors[:, np.newaxis] * notes)
        bias_gradient = fm.mean(errors)

        self.weights -= self.learning_rate * gradients
        self.bias -= self.learning_rate * bias_gradient

    def inverse_time_decay(
        self, initial_lr: float, decay_rate: float, iteration: int
    ) -> float:
        return initial_lr / (1 + decay_rate * iteration)

    def compute(self, batch: np.ndarray, y_batch: np.ndarray) -> float:
        predictions = fm.sigmoid(batch @ self.weights + self.bias)
        errors = predictions - y_batch
        self.gradient_descent(errors, batch)
        return fm.mean(self.cost_function(y_batch, predictions))


class Batch(LogReg):
    def __init__(
        self,
        notes: np.ndarray,
        house: str,
        students_houses: np.ndarray,
    ):
        super().__init__(notes, house, students_houses)
        self.LOOPS = 10000
        self.stats = TrainingStats(house, self.LOOPS)
        self.y = (students_houses == house).astype(float)
        self.learning_rate = 0.05

    def predict(self) -> tuple[np.ndarray, float]:
        for i in range(self.LOOPS):
            loss = self.compute(self.notes, self.y)
            self.stats.records[i] = loss

        predictions = fm.sigmoid(self.notes @ self.weights + self.bias)
        loss = fm.mean(self.cost_function(self.y, predictions))
        print(loss)

        self.stats.plot()

        return self.weights, self.bias


class MiniBatch(LogReg):
    def __init__(
        self,
        notes: np.ndarray,
        house: str,
        students_houses: np.ndarray,
    ):
        super().__init__(notes, house, students_houses)
        self.decay_rate: float = 0.5
        self.LOOPS = 10000
        self.stats = TrainingStats(house, self.LOOPS)
        self.y = (students_houses == house).astype(float)
        self.max = notes.shape[0]
        self.BATCH_SIZE = 50

        self.EPOCH = self.max / self.BATCH_SIZE

        print(self.EPOCH)

    def predict(self) -> tuple[np.ndarray, float]:
        for i in range(self.LOOPS):
            index_low = random.randrange(self.max - self.BATCH_SIZE)
            batch = self.notes[index_low : index_low + self.BATCH_SIZE, :]
            y_batch = self.y[index_low : index_low + self.BATCH_SIZE]

            loss = self.compute(batch, y_batch)
            self.stats.records[i] = loss
            if i % (self.EPOCH * 30) == 0:
                self.learning_rate *= self.decay_rate

        predictions = fm.sigmoid(self.notes @ self.weights + self.bias)
        loss = fm.mean(self.cost_function(self.y, predictions))
        print(loss)

        self.stats.plot()

        return self.weights, self.bias


class Stochastic(LogReg):
    def __init__(
        self,
        notes: np.ndarray,
        house: str,
        students_houses: np.ndarray,
    ):
        super().__init__(notes, house, students_houses)
        self.decay_rate: float = 0.01
        self.learning_rate = 0.01
        self.LOOPS = 10000
        self.stats = TrainingStats(house, self.LOOPS)
        self.y = (students_houses == house).astype(float)
        self.max = notes.shape[0]

    def predict(self) -> tuple[np.ndarray, float]:
        for i in range(self.LOOPS):
            index_low = random.randrange(self.max - 1)
            batch = self.notes[index_low : index_low + 1, :]
            y_batch = self.y[index_low : index_low + 1]

            loss = self.compute(batch, y_batch)
            self.stats.records[i] = loss
            self.learning_rate = self.inverse_time_decay(
                0.1, self.decay_rate, i
            )

        predictions = fm.sigmoid(self.notes @ self.weights + self.bias)
        loss = fm.mean(self.cost_function(self.y, predictions))
        print(loss)

        self.stats.plot()

        return self.weights, self.bias

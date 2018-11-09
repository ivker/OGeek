class Gain:
    @staticmethod
    def f1(acc: float, recall: float) -> float:
        return 2 * (acc * recall) / (acc + recall)

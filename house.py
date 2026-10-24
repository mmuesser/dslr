import math


class House:
    def __init__(self, name: str, color: str):
        self.name: str = name
        self.color: str = color
        self.markbook: dict[str, list] = {}

    def initMarkBook(self, courses: list[str]):
        for course in courses:
            self.markbook[course] = []

    def addMark(self, course: str, mark: float):
        if not math.isnan(mark):
            book = self.markbook[course]
            book.append(mark)

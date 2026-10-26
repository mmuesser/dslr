import math


class House:
    def __init__(self, name: str, color: str):
        self.name: str = name
        self.color: str = color
        self.markbook: dict[str, list] = {}
        self.rawmarks: dict[str, list] = {}

    def initMarkBook(self, courses: list[str]):
        for course in courses:
            self.markbook[course] = []

    def initRawMarks(self, courses: list[str]):
        for course in courses:
            self.rawmarks[course] = []

    def addMark(self, course: str, mark: float):
        if not math.isnan(mark):
            book = self.markbook[course]
            book.append(mark)

    def addRawMark(self, course: str, mark: float):
        book = self.rawmarks[course]
        book.append(mark)

class House:
    def __init__(self, name: str, color: str):
        self.name: str = name
        self.color: str = color
        self.markbook: dict[str, list[float]] = {}
        self.rawmarks: dict[str, list[float]] = {}

    def initMarkBook(self, courses: list[str]):
        for course in courses:
            self.markbook[course] = []

    def addMark(self, course: str, mark: float):
        book = self.markbook[course]
        book.append(mark)

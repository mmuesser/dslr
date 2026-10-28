import pandas
import math

from house import House


class School:
    def __init__(self):
        self.houses: dict[str, House] = {
            "Gryffindor": House("Gryffindor", "red"),
            "Hufflepuff": House("Hufflepuff", "orange"),
            "Ravenclaw": House("Ravenclaw", "blue"),
            "Slytherin": House("Slytherin", "green"),
        }

    def insertData(self, file_name: str):

        datas = self.extractData(file_name)

        self.min_values: dict[str, float] = {}
        self.max_values: dict[str, float] = {}
        for house in self.houses.values():
            house.initMarkBook(self.courses)

        for course in self.courses:
            self.insertCourse(course, datas)

    def extractData(self, file_name):
        datas = pandas.read_csv(file_name)
        num_data = datas.select_dtypes(include=[float, int])
        num_data.drop(columns=["Index"], inplace=True)
        self.courses = list(num_data.columns)
        return datas

    def insertCourse(self, course: str, datas):
        minimum: float = math.nan
        maximum: float = math.nan

        for _, row in datas.iterrows():
            name: str = str(row["Hogwarts House"])
            house: House = self.houses[name]
            mark: float = float(row.get(course))  # type: ignore[arg-type]

            house.addMark(course, mark)
            if not math.isnan(mark):
                minimum = min(mark, minimum)
                maximum = max(mark, maximum)
        self.min_values[course] = minimum
        self.max_values[course] = maximum

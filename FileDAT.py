import numpy as np
from IPython.display import display
import pandas as pd


class FileDAT:

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None

        self._header = pd.DataFrame
        self._x = np.array([])
        self._y = np.array([])

        self.content = ""
        self.name = ""
        self.num_rows = 0
        self.num_attr = 0

    def _extract_data(self, raw_data):
        data = np.array([x.replace("\n", "").split(",") for x in raw_data])
        self._x = data[:, :-1]
        # Para ser possível concatenar (hstack) x e y adicionar uma dimensão ao y
        # To concat (hstack) "x" and "y" it is required that they have the same number of dimensions
        # therefore add one dimension to "y" to be a 2D array
        self._y = np.expand_dims(data[:, -1], 1)

    def _extract_header(self, raw_header):
        self._header = pd.DataFrame(raw_header, columns=["Attribute Name", "Domain", "Range"])

    def get_data(self):
        if not (self.file and self.content) or (self.num_rows == 0):
            raise AttributeError("File is not open")

        return np.hstack((self._x, self._y))

    def open(self):
        self.file = open(self.filepath, 'r')
        self.content = self.file.readlines()

    def interpret(self):
        if not (self.file and self.content):
            raise AttributeError("File is not open")

        self.name = self.content[0].split()[-1]
        raw_header: list[list[str]] = []

        for i, line in enumerate(self.content):
            if ("@attribute" in line):
                self.num_attr += 1
                line_aux = line.replace(", ", ",").replace("@attribute", "").split()

                if (len(line_aux) < 3):
                    line_aux.insert(1, "categorical")

                raw_header.append(line_aux)

            if ("@data" in line):
                self.num_rows = len(self.content) - i - 1
                self._extract_data(self.content[i + 1:])
                break

        self._extract_header(raw_header)
        self.file.close()

    def info(self):
        if (self.num_attr == 0):
            raise ValueError("File was not interpreted")

        print(f"Name of the dataset: {self.name}")
        display(self._header)
        print(f"Number of features: {self.num_attr - 1}")
        print(f"Number of instances: {self.num_rows}")
        print(f"Number of classes: {len(np.unique(self._y))}")

    def to_csv(self, path: str):
        if (self.num_attr == 0):
            raise ValueError("File was not interpreted")

        csv_format = np.vstack((np.array(self._header["Attribute Name"]), self.get_data()))
        np.savetxt(path, np.array(csv_format), delimiter=",", fmt="%s")

        return csv_format


    def get_header(self):
        return self._header

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y


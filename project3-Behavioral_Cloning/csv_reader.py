import csv
from collections import namedtuple

CsvRow = namedtuple("CsvRow", ["img_center", "img_left", "img_right", "steering", "throttle", "brake", "speed"])

class CsvReader:

    def __init__(self, path):
        self.reader = csv.reader(open(path))

    def line_to_row(self, line):
        return CsvRow(img_center=line[0], img_left=line[1], img_right=line[2],
                      steering=line[3], throttle=line[4], brake=line[5], speed=line[6])

    def __AdjustImagePathTo(self, img_path, new_path):
        basename=img_path.split('/')[-1]
        return "%s/%s" % (new_path, basename)

    def next(self):
        return self.line_to_row(next(self.reader))

class AdjustableCsvReader(CsvReader):

    def __init__(self, path, adjustable_path):
        super().__init__(path)
        self.adjustable_path = adjustable_path

    def __AdjustImagePathTo(self, img_path, new_path):
        basename=img_path.split('/')[-1]
        return "%s/%s" % (new_path, basename)

    def next(self):
        line = next(self.reader)
        for i in range(3):
            line[i] = self.__AdjustImagePathTo(line[i], self.adjustable_path)
        return self.line_to_row(line)

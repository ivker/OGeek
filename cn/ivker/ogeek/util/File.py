from functools import reduce
import numpy as np
import logging

logger = logging.getLogger("File")


class File:

    @staticmethod
    def read1DArray(path: str, type=None):
        return np.asarray(File.readList(path), dtype=type)

    @staticmethod
    def read2DArray(path: str, type=None):
        return np.asarray(File.readListInList(path), dtype=type)

    @staticmethod
    def readList(path: str, encoding='utf-8') -> list:
        with open(path, "r", encoding=encoding) as f:
            records = list(map(lambda x: x.strip(), f.readlines()))
            logger.info('Read ' + str(len(records)) + ' record(s) from ' + path)
            return records

    @staticmethod
    def readListInList(path: str, encoding='utf-8') -> list:
        with open(path, "r", encoding=encoding) as f:
            data = f.readlines()
            records = []
            for item in data:
                record = item.strip().split("\t")
                records.append(record)
            logger.info('Read ' + str(len(records)) + ' record(s) from ' + path)
            return records

    @staticmethod
    def writeList(path: str, data: list, encoding='utf-8'):
        with open(path, "w", encoding=encoding) as f:
            f.writelines([str(line) + '\n' for line in data])

    @staticmethod
    def writeListInList(path: str, data: list, encoding='utf-8'):
        with open(path, "w", encoding=encoding) as f:
            for item in data:
                f.writelines(str(reduce(lambda x, y: str(x) + "\t" + str(y), item)) + "\n")

    @staticmethod
    def writelines(path: str, data: str, encoding='utf-8'):
        with open(path, "w", encoding=encoding) as f:
            f.writelines(data)

    @staticmethod
    def writeDict(path: str, d: dict, encoding='utf-8'):
        with open(path, "w", encoding=encoding) as f:
            for key in d:
                f.writelines(key + ":" + str(d[key]) + "\n")

    @staticmethod
    def readDict(path: str, encoding='utf-8'):
        with open(path, "r", encoding=encoding) as f:
            data = f.readlines()
            d = {}
            for r in data:
                kv = r.split(":")
                d.setdefault(kv[0], kv[1])

            return d

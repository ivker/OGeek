from cn.ivker.ogeek.util.File import File


class Statistician:

    def __init__(self):
        self.statistics = {}

    def get(self, x: list):
        pass

    def dump(self, path):
        # joblib.dump(self, path)
        File.writeDict(path, self.statistics)

    @staticmethod
    def load(path):
        # return joblib.load(path)
        stn = Statistician()
        stn.statistics = File.readDict(path)
        return stn

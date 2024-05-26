import pandas as pd

from EmotionsDataset import get_path

class GoEmotionsDataset():
    def __init__(self, part1, part2, part3):
        p1_csv = pd.read_csv(get_path(part1))
        p2_csv = pd.read_csv(get_path(part2))
        p3_csv = pd.read_csv(get_path(part3))

        self.csv = pd.concat([p1_csv, p2_csv, p3_csv])

    def out_csv(self):
        for i in self.csv:
            print(i)

if __name__ == '__main__':
    go_emotions = GoEmotionsDataset("data/go_emotions_1.csv",
                                    "data/go_emotions_2.csv",
                                    "data/go_emotions_3.csv")

    go_emotions.out_csv()
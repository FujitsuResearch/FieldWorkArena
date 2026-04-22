class FWA_taskCategory:
    def __init__(self, category):
        self.category = category
        self.totalTaskNum = 0
        self.correctNum = 0
        self.correctRate = 0.0
        self.totalScore = 0.0

    def __str__(self):
        return f"Task category {self.category} : input totalTaskNum {self.totalTaskNum}, correctNum {self.correctNum}, correctRate {self.correctRate}, totalScore {self.totalScore}"

    def __repr__(self):
        return f"Task category {self.category} : input totalTaskNum {self.totalTaskNum}, correctNum {self.correctNum}, correctRate {self.correctRate}, totalScore {self.totalScore}"

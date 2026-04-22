
class FWA_task:
    def __init__(self, id, data, query, answer, eval_func):
        self.id = id
        self.data = data
        self.query = query
        self.answer = answer
        self.eval_func = eval_func

    def __str__(self):
        return f"Task id {self.id} : input data {self.data}, query {self.query}, answer {self.answer}"

    def __repr__(self):
        return f"Task id {self.id} : input data {self.data}, query {self.query}, answer {self.answer}"


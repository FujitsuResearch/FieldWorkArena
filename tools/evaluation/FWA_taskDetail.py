class FWA_taskDetail:
    def __init__(self, id , query, answer, groundtruth, eval_func, result, memo, input_tokens, output_tokens):
        self.id = id
        self.query = query
        self.answer = answer
        self.groundtruth = groundtruth
        self.eval_func = eval_func
        self.result = result
        self.memo = memo
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __str__(self):
        return f"{self.id}\t {self.query}\t {self.answer}\t {self.groundtruth}\t {self.eval_func}\t {self.result}\t {self.memo}\t {self.input_tokens}\t {self.output_tokens}"

    def __repr__(self):
        return f"{self.id}\t {self.query}\t {self.answer}\t {self.groundtruth}\t {self.eval_func}\t {self.result}\t {self.memo}\t {self.input_tokens}\t {self.output_tokens}"
    
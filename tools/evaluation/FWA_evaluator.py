import json
import os
import re

from FWA_task import FWA_task
from FWA_taskCategory import FWA_taskCategory
from FWA_taskDetail import FWA_taskDetail
import metrics.automatic.automatic_evaluation as auto_eval

from config import Config

from FWA_evaluator_logger import set_logger, getLogger
set_logger()
logger = getLogger(__name__)


class FWA_evaluator:
    def __init__(self, target, gt):
        self.target = target
        self.gt = gt

    def evaluate_func(self, ref, pred, query, func):
        score = 0.0
        input_tokens = 0
        output_tokens = 0
        match func:
            case "fuzzy_match":
                score, memo, input_tokens, output_tokens = auto_eval.llm_fuzzy_match(pred, ref, query)
                logger.info(" ==> fuzzy_match, score: {}".format(score))
            case "exact_match":
                score, memo = auto_eval.exact_match(ref, pred)
                logger.info(" ==> exact_match, score: {}".format(score))
            case "must_include":
                score, memo = auto_eval.must_include(ref, pred)
                logger.info(" ==> must_include, score: {}".format(score))
            case "must_exclude":
                score, memo = auto_eval.must_exclude(ref, pred)
                logger.info(" ==> must_exclude, score: {}".format(score))
            case "json_match":
                score, memo, input_tokens, output_tokens = auto_eval.json_match(pred, ref, query)
                logger.info(" ==> json_match, score: {}".format(score))
            case "numerical_match":
                score, memo, input_tokens, output_tokens = auto_eval.numerical_match(pred, ref, query)
                logger.info(" ==> numerical_match, score: {}".format(score))   
        # print("score: ", score)
        return score, memo, input_tokens, output_tokens

    def extract_task_category(self, taskId):
        task_category = taskId[:-5]
        return task_category

    def evaluate(self):
        matching_elements = self.find_matching_elements()
        correct_ids = []

        task_category = {}
        task_category["total"] = FWA_taskCategory("total")

        task_detail = []

        proceed_num = 0
        for t, g in matching_elements:

            # find the task category
            category = self.extract_task_category(g.id)
            if category not in task_category.keys():
                task_category[category] = FWA_taskCategory(category)

            # evaluate the task
            logger.info("id: {}".format(g.id))
            task_category["total"].totalTaskNum += 1
            task_category[category].totalTaskNum += 1
            score, memo, input_tokens, output_tokens = self.evaluate_func(g.answer, t.answer, g.query, g.eval_func)
            #task_detail.append(FWA_taskDetail(g.id, g.query, t.answer, g.answer, g.eval_func, True if score > 0.5 else False))
            task_detail.append(FWA_taskDetail(g.id, g.query, t.answer, g.answer, g.eval_func, score, memo, input_tokens, output_tokens))

            if score > 0.5:
                task_category["total"].correctNum += 1
                task_category[category].correctNum += 1
                correct_ids.append(t.id)
            task_category["total"].totalScore += score
            task_category[category].totalScore += score

            task_category["total"].correctRate = "{:.4f}".format((task_category["total"].correctNum)/task_category["total"].totalTaskNum)
            task_category[category].correctRate = "{:.4f}".format((task_category[category].correctNum)/task_category[category].totalTaskNum)
            
            proceed_num += 1
            print("proceed: ", "{:.2f}".format((proceed_num*100)/len(matching_elements)), "%  [", proceed_num, "/", len(matching_elements), "]")

        return task_category, task_detail

    def find_matching_elements(self):
        matching_elements = []
        for t in self.target:
            if t.id == "":
                continue
            for g in self.gt:
                if t.id == g.id:
                    matching_elements.append((t, g))
                    break
        return matching_elements


def start(target_file, groundtruth_file, result_file):
    logger.info("------------------")
    logger.info("start evaluation")
    logger.info("target: {}".format(target_file))
    logger.info("groundtruth: {}".format(groundtruth_file))
    logger.info("result: {}".format(result_file))

    # input target files
    target_task_list = []

    file_list = []
    file_list = recursive_file_search(target_file, file_list, ".log")
    for task_file in file_list:
        target_data = read_target(task_file)
        if len(target_data) == 0:
            logger.info("{} has no <id> and <answer>.".format(task_file))
            continue
        target_task = parse_target(target_data)
        target_task_list.append(target_task)

    # input GT files
    file_list = []
    file_list = recursive_file_search(groundtruth_file, file_list, ".json")
    GT_task_list = []
    for task_file in file_list:
        GT_data = read_GT(task_file)
        if len(GT_data) == 0:
            logger.info("{} has no <id> and <answer>.".format(task_file))
            continue
        GT_task_list = parse_GT(GT_data, GT_task_list)
        # target_task_list.append(target_task)

    # evaluate
    evaluator = FWA_evaluator(target_task_list, GT_task_list)
    result_summary, result_detail = evaluator.evaluate()

    # output the summary of the evaluation
    result_json = convert_to_json(result_summary)
    json.dump(result_json, open(result_file, 'w'), indent=4, separators=(',', ':'))

    # output the detail of the evaluation
    with open(result_file + Config.DETAIL_RESULT_SUFFIX, "w", encoding="utf-8-sig") as f:
        for result in result_detail:
            #f.write(f"{result.id}\t\"{result.query.replace("\t", " ")}\"\t\"{result.answer.replace("\t", " ")}\"\t\"{result.groundtruth.replace("\t", " ")}\"\t{result.eval_func},{result.result}\n")
            f.write(str(result) + "\n")


def recursive_file_search(root_dir, file_list, extention):
    for (path, dir, files) in os.walk(root_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extention:
                file_list.append(path + "/" + filename)
    return file_list


def read_target(filepath):
    ENCODE_TYPE = 'utf-8', 'utf-8-sig', 'shift-jis', 'cp932'

    # read the execution log files
    # print(filepath)
    target_data = []
    for c in ENCODE_TYPE:
        try:
            with open(os.path.join(filepath), 'r', encoding=c) as f:
                text = f.read()

                # search <id>-</id> pair and set "id"
                prog = re.compile(r"<id>(.*?)</id>", re.MULTILINE | re.DOTALL)
                search = re.search(prog, text)
                if search is None:
                    break
                id = re.sub(r"<(.*?)>", "", search.group(0))
                id = re.sub(r"\n", "", id)

                # search <answer>-</answer> pair and set "answer"
                prog = re.compile(r"<answer>(.*?)</answer>", re.MULTILINE | re.DOTALL)
                search = re.search(prog, text)
                if search is None:
                    break
                answer = re.sub(r"<(.*?)>", "", search.group(0))
                answer = re.sub(r"\n", "", answer)

                target_data.append(id)
                target_data.append(answer)
                break
        except UnicodeDecodeError:
            c = "unknown"
            continue

    logger.debug("{} is encode type : {}.".format(filepath, c))
    return target_data


def read_GT(filepath):

    # read the json files
    # print(filepath)
    with open(os.path.join(filepath), 'r', encoding='utf-8-sig') as f:
        GT_data = json.load(f)

    return GT_data


def parse_target(data):

    # id : data[0] = <id>fieldworkarena.XXXXXXX</id>
    # id = data[0][19:len(data[0])-6]
    id = data[0][15:]

    # answer : data[1] = <answer>XXXXXXX</answer>
    # answer = data[1][8:len(data[1])-9]
    answer = data[1]

    task = FWA_task(id, "", "", answer, "")

    return task


def parse_GT(data, task_list):
    for task in data:
        task_list.append(FWA_task(task["id"], task["input_data"], task["conversations"][0]["value"], task["conversations"][1]["value"], task["eval_func"]))

    return task_list


def convert_to_json(result):
    result_json = {}
    for key, value in result.items():
        result_json[key] = {}
        result_json[key]["totalTaskNum"] = value.totalTaskNum
        result_json[key]["correctNum"] = value.correctNum
        result_json[key]["correctRate"] = value.correctRate
        result_json[key]["totalScore"] = value.totalScore

    return result_json
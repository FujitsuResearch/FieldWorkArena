import sys
import FWA_evaluator


def main(target, groundtruth, result):
    FWA_evaluator.start(target, groundtruth, result)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run-evaluation.py <target file> <ground truth file> <result file>")
        sys.exit(1)

    target = sys.argv[1]
    groundtruth = sys.argv[2]
    result = sys.argv[3]

    main(target, groundtruth, result)

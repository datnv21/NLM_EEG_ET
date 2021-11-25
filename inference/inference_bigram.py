import os
import sys
sys.path.append("models")

import argparse
from bigram import BigramModel


def parse_args(args):
    parser = argparse.ArgumentParser(description="process arguments")
    parser.add_argument("--sent", help="sentence", dest="sent", type=str, required=True)
    parser.add_argument("--threshold", help="threshold", dest="threshold", type=float, default=0.12)
    print(vars(parser.parse_args(args)))
    
    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    input_w = args.sent.split("_")[-1]
    threshold = args.threshold

    bigram_ins = BigramModel()
    model = bigram_ins.bigram_wordcount()

    recommend_ws = []

    for w in model[input_w]:
        if (model[input_w][w] > threshold) and w is not None and len(w): recommend_ws.append(w)

    print(recommend_ws)
    return 1


if __name__=="__main__":
    main()

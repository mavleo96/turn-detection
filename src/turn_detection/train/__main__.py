from transformers import HfArgumentParser

from .trainer import EoTTrainingArguments, train


def parse_args() -> EoTTrainingArguments:
    """Command line -> EoTTrainingArguments."""
    (args,) = HfArgumentParser(EoTTrainingArguments).parse_args_into_dataclasses()
    return args


if __name__ == "__main__":
    train(parse_args())

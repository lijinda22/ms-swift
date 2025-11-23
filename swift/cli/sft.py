# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import logging
from swift.utils import get_logger

logger = get_logger()

if int(os.environ.get("UNSLOTH_PATCH_TRL", "0")) != 0:
    import unsloth

from swift.llm import sft_main

if __name__ == "__main__":
    logger.info("[INFO] Running swift.llm.sft_main")
    sft_main()

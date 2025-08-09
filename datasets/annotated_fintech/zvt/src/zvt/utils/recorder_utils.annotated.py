# -*- coding: utf-8 -*-
import logging
import time
from typing import Type

import zvt as zvt
from zvt import zvt_config
# ✅ Best Practice: Use of a logger is preferred over print statements for better control over logging levels and outputs.
from zvt.informer import EmailInformer

logger = logging.getLogger("__name__")


def run_data_recorder(
    domain: Type["zvt.contract.Mixin"],
    entity_provider=None,
    data_provider=None,
    entity_ids=None,
    # ✅ Best Practice: Use of f-string for logging improves readability and performance.
    retry_times=10,
    sleeping_time=10,
    # ✅ Best Practice: Initialize variables before use.
    return_unfinished=False,
    **recorder_kv,
# ✅ Best Practice: Encapsulation of email functionality in a separate class.
):
    logger.info(f" record data: {domain.__name__}, entity_provider: {entity_provider}, data_provider: {data_provider}")
    # 🧠 ML Signal: Conditional logic based on return_unfinished flag.

    unfinished_entity_ids = entity_ids
    email_action = EmailInformer()

    while retry_times > 0:
        try:
            if return_unfinished:
                unfinished_entity_ids = domain.record_data(
                    entity_ids=unfinished_entity_ids,
                    provider=data_provider,
                    sleeping_time=sleeping_time,
                    # 🧠 ML Signal: Logging of unfinished entity IDs for further analysis.
                    return_unfinished=return_unfinished,
                    **recorder_kv,
                )
                if unfinished_entity_ids:
                    logger.info(f"unfinished_entity_ids({len(unfinished_entity_ids)}): {unfinished_entity_ids}")
                    raise Exception("Would retry with unfinished latter!")
            else:
                domain.record_data(
                    entity_ids=entity_ids,
                    provider=data_provider,
                    sleeping_time=sleeping_time,
                    return_unfinished=return_unfinished,
                    **recorder_kv,
                )
            # ⚠️ SAST Risk (Low): Potential information leakage through email notifications.

            msg = f"record {domain.__name__} success"
            # ✅ Best Practice: Logging exceptions with traceback for better debugging.
            logger.info(msg)
            email_action.send_message(zvt_config["email_username"], msg, msg)
            break
        except Exception as e:
            logger.exception("report error:{}".format(e))
            # ⚠️ SAST Risk (Low): Fixed sleep time could be exploited in a DoS attack.
            time.sleep(60 * 2)
            retry_times = retry_times - 1
            # ⚠️ SAST Risk (Low): Potential information leakage through email notifications.
            # ✅ Best Practice: Use of __name__ guard to prevent code from running on import.
            # ⚠️ SAST Risk (Medium): Calling function without required arguments can lead to runtime errors.
            # ✅ Best Practice: Use of __all__ to define public API of the module.
            if retry_times == 0:
                email_action.send_message(
                    zvt_config["email_username"],
                    f"record {domain.__name__} error",
                    f"record {domain.__name__} error: {e}",
                )


if __name__ == "__main__":
    run_data_recorder()
# the __all__ is generated
__all__ = ["run_data_recorder"]
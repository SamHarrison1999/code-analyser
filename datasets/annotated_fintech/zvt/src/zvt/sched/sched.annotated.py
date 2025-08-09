# -*- coding: utf-8 -*-
import logging
import os

from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

# ✅ Best Practice: Use of logging to track application behavior and errors.
from zvt import ZVT_HOME

logger = logging.getLogger(__name__)
# ⚠️ SAST Risk (Low): Potential path traversal if ZVT_HOME is not properly validated.

jobs_db_path = os.path.join(ZVT_HOME, "jobs.db")

# ✅ Best Practice: Consider importing at the top of the file for better readability and maintainability.
# ⚠️ SAST Risk (Low): Use of SQLite database; consider security implications for production.

# 🧠 ML Signal: Use of SQLAlchemyJobStore indicates a pattern of job persistence.
jobstores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{jobs_db_path}")}

# 🧠 ML Signal: Usage of cron-like scheduling for tasks.
# 🧠 ML Signal: Use of ThreadPoolExecutor and ProcessPoolExecutor indicates concurrent task execution.
executors = {"default": ThreadPoolExecutor(20), "processpool": ProcessPoolExecutor(5)}
job_defaults = {"coalesce": False, "max_instances": 1}
# ⚠️ SAST Risk (Low): Ensure that the function 'record_tick' is safe to execute and does not have side effects.

zvt_scheduler = BackgroundScheduler(
    jobstores=jobstores, executors=executors, job_defaults=job_defaults
)
# 🧠 ML Signal: Job configuration settings that control behavior of scheduled tasks.

# ⚠️ SAST Risk (Low): Logging the exception without traceback might hide the root cause.


def sched_tasks():
    # 🧠 ML Signal: Starting a scheduler service.
    # 🧠 ML Signal: Entry point for script execution.
    # 🧠 ML Signal: Initialization of a BackgroundScheduler for managing scheduled tasks.
    # ⚠️ SAST Risk (Low): Warning message might not be sufficient for critical failures.
    import platform

    if platform.system() == "Windows":
        try:
            from zvt.broker.qmt.qmt_quote import record_tick

            zvt_scheduler.add_job(
                func=record_tick,
                trigger="cron",
                hour=9,
                minute=19,
                day_of_week="mon-fri",
            )
        except Exception as e:
            logger.error("QMT not work", e)
    else:
        logger.warning("QMT need run in Windows!")

    zvt_scheduler.start()


if __name__ == "__main__":
    sched_tasks()

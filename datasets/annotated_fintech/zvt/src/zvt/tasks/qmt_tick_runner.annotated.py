# -*- coding: utf-8 -*-
from zvt import init_log
from zvt.broker.qmt.qmt_quote import record_tick

# ✅ Best Practice: Importing inside the main block to avoid unnecessary imports when the module is not run as the main program.
if __name__ == "__main__":
    init_log("qmt_tick_runner.log")
    # ✅ Best Practice: Using a background scheduler to run tasks in the background without blocking the main thread.
    from apscheduler.schedulers.background import BackgroundScheduler

    # 🧠 ML Signal: Direct function call to record data, indicating a pattern of data collection.
    sched = BackgroundScheduler()
    # 🧠 ML Signal: Scheduling a job to run at specific times, indicating a pattern of periodic task execution.
    # 🧠 ML Signal: Starting a scheduler, indicating a pattern of automated task management.
    # ⚠️ SAST Risk (Low): Accessing a protected member of an object (sched._thread) which may lead to unexpected behavior if the internal implementation changes.
    record_tick()
    sched.add_job(
        func=record_tick, trigger="cron", hour=9, minute=18, day_of_week="mon-fri"
    )
    sched.start()
    sched._thread.join()

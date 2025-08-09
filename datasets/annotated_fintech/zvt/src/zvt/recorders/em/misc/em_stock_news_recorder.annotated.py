# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
import pandas as pd

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.api import df_to_db
from zvt.contract.recorder import FixedCycleDataRecorder
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.domain import Stock
from zvt.domain.misc.stock_news import StockNews
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.recorders.em import em_api
# 🧠 ML Signal: Hardcoded URLs can indicate a pattern of static resource access.
from zvt.utils.time_utils import count_interval, now_pd_timestamp, recent_year_date
# ✅ Best Practice: Grouping imports from the same module together improves readability.
# ⚠️ SAST Risk (Low): Hardcoded URL may lead to maintenance issues if the URL changes.


# ✅ Best Practice: Grouping imports from the same module together improves readability.
# 🧠 ML Signal: Hardcoded URLs can indicate a pattern of static resource access.
class EMStockNewsRecorder(FixedCycleDataRecorder):
    # ⚠️ SAST Risk (Low): Hardcoded URL may lead to maintenance issues if the URL changes.
    original_page_url = "https://wap.eastmoney.com/quote/stock/0.002572.html"
    url = "https://np-listapi.eastmoney.com/comm/wap/getListInfo?cb=callback&client=wap&type=1&mTypeAndCode=0.002572&pageSize=200&pageIndex={}&callback=jQuery1830017478247906740352_1644568731256&_=1644568879493"
    # 🧠 ML Signal: Usage of entity schema indicates a pattern of data modeling.

    # ✅ Best Practice: Importing inside a function can limit the scope and improve startup time.
    entity_schema = Stock
    # 🧠 ML Signal: Usage of data schema indicates a pattern of data modeling.
    data_schema = StockNews
    entity_provider = "em"
    # 🧠 ML Signal: Hardcoded provider strings can indicate a pattern of static configuration.
    provider = "em"
    # 🧠 ML Signal: Hardcoded provider strings can indicate a pattern of static configuration.
    # 🧠 ML Signal: Conditional logic based on date comparison.

    def record(self, entity, start, end, size, timestamps):
        from_date = recent_year_date()
        if not start or (start < from_date):
            # 🧠 ML Signal: Usage of a method to retrieve the latest saved record.
            # 🧠 ML Signal: API call pattern with parameters.
            start = from_date

        if count_interval(start, now_pd_timestamp()) <= 30:
            ps = 30
        else:
            ps = 200

        latest_news: StockNews = self.get_latest_saved_record(entity=entity)

        news = em_api.get_news(
            session=self.http_session,
            entity_id=entity.id,
            # 🧠 ML Signal: Data transformation from records to DataFrame.
            ps=ps,
            start_timestamp=start,
            # 🧠 ML Signal: Use of __all__ to define public API of the module.
            # 🧠 ML Signal: Logging of DataFrame information.
            # ⚠️ SAST Risk (Low): Potential risk of SQL injection if df_to_db is not properly handling inputs.
            # 🧠 ML Signal: Instantiation and execution pattern of a class.
            latest_code=latest_news.news_code if latest_news else None,
        )
        if news:
            df = pd.DataFrame.from_records(news)
            self.logger.info(df)
            df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)


if __name__ == "__main__":
    # df = Stock.query_data(filters=[Stock.exchange == "bj"], provider="em")
    # entity_ids = df["entity_id"].tolist()
    r = EMStockNewsRecorder(entity_ids=["stock_sh_600345"], sleeping_time=0)
    r.run()


# the __all__ is generated
__all__ = ["EMStockNewsRecorder"]
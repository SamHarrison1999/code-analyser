# -*- coding: utf-8 -*-

from sqlalchemy.sql.expression import text

from zvt.contract import Exchange
from zvt.contract.api import df_to_db
from zvt.contract.recorder import Recorder
from zvt.domain import Stock

# 🧠 ML Signal: Inheritance from a base class, indicating a pattern of extending functionality
from zvt.recorders.em import em_api
from zvt.utils.pd_utils import pd_is_not_null

# 🧠 ML Signal: Class attribute indicating a constant or configuration setting

# 🧠 ML Signal: Iterating over a list of exchanges indicates a pattern of processing multiple data sources.


# 🧠 ML Signal: Class attribute indicating a schema or data structure being used
class EMStockRecorder(Recorder):
    # 🧠 ML Signal: API call to fetch tradable list, indicating data retrieval pattern.
    provider = "em"
    data_schema = Stock
    # ✅ Best Practice: Check if DataFrame is not null before processing.

    def run(self):
        # 🧠 ML Signal: Iterating over DataFrame rows to process each item.
        for exchange in [Exchange.sh, Exchange.sz, Exchange.bj]:
            df = em_api.get_tradable_list(entity_type="stock", exchange=exchange)
            # ✅ Best Practice: Destructuring assignment for clarity.
            # df_delist = df[df["name"].str.contains("退")]
            if pd_is_not_null(df):
                for item in df[["id", "name"]].values.tolist():
                    # ⚠️ SAST Risk (High): SQL injection risk due to string formatting in SQL query.
                    id = item[0]
                    name = item[1]
                    # 🧠 ML Signal: Executing SQL command indicates database interaction pattern.
                    sql = text(f'update stock set name = "{name}" where id = "{id}"')
                    # 🧠 ML Signal: Committing transaction to database.
                    # 🧠 ML Signal: Logging information, indicating a pattern of monitoring or debugging.
                    # 🧠 ML Signal: Saving DataFrame to database, indicating data persistence pattern.
                    # ✅ Best Practice: Use of main guard to ensure code only runs when script is executed directly.
                    # 🧠 ML Signal: Instantiating and running a recorder object, indicating a pattern of task execution.
                    # ✅ Best Practice: Use of __all__ to define public API of the module.
                    self.session.execute(sql)
                    self.session.commit()
            self.logger.info(df)
            df_to_db(
                df=df,
                data_schema=self.data_schema,
                provider=self.provider,
                force_update=self.force_update,
            )


if __name__ == "__main__":
    recorder = EMStockRecorder()
    recorder.run()


# the __all__ is generated
__all__ = ["EMStockRecorder"]

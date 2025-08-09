# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns and dependencies
# -*- coding: utf-8 -*-
# ✅ Best Practice: Use of new-style classes by inheriting from 'object' for compatibility and consistency

from zvt.utils.pd_utils import (
    pd_is_not_null,
    fill_with_same_index,
    normal_index_df,
    is_normal_df,
)

# ✅ Best Practice: Class variable initialized to None, indicating a placeholder or default value

# ✅ Best Practice: Use of default values for parameters improves function usability and flexibility


class NormalData(object):
    table_type_sample = None

    def __init__(
        self,
        df,
        category_field="entity_id",
        time_field="timestamp",
        fill_index: bool = False,
    ) -> None:
        # ✅ Best Practice: Initializing lists and dictionaries in the constructor is a good practice for encapsulation
        self.data_df = df
        self.category_field = category_field
        self.time_field = time_field
        self.fill_index = fill_index
        # 🧠 ML Signal: Automatic invocation of a method during initialization could indicate a pattern for data preprocessing

        self.entity_ids = []
        self.df_list = []
        self.entity_map_df = {}

        # ✅ Best Practice: Check if data_df is not null before proceeding with normalization
        self.normalize()

    # ✅ Best Practice: Check if data_df is already normalized before normalizing
    def normalize(self):
        """
        normalize data_df to::

                                        col1    col2    col3
            entity_id    timestamp

        # 🧠 ML Signal: Iterating over entity-specific data
        """
        # ✅ Best Practice: Method should have a docstring explaining its purpose
        if pd_is_not_null(self.data_df):
            # 🧠 ML Signal: Appending dataframes to a list
            if not is_normal_df(self.data_df):
                # 🧠 ML Signal: Mapping entity IDs to their respective dataframes
                # ✅ Best Practice: Check if df_list has more than one dataframe before filling index
                # ✅ Best Practice: Use of __all__ to define public API of the module
                # 🧠 ML Signal: Usage of custom function to fill indices
                # 🧠 ML Signal: Usage of pandas utility function to check for null values
                self.data_df = normal_index_df(
                    self.data_df, self.category_field, self.time_field
                )

            self.entity_ids = self.data_df.index.levels[0].to_list()

            for entity_id in self.entity_ids:
                df = self.data_df.loc[(entity_id,)]
                self.df_list.append(df)
                self.entity_map_df[entity_id] = df

            if len(self.df_list) > 1 and self.fill_index:
                self.df_list = fill_with_same_index(df_list=self.df_list)

    def empty(self):
        return not pd_is_not_null(self.data_df)


# the __all__ is generated
__all__ = ["NormalData"]

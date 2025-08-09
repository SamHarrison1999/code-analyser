# 🧠 ML Signal: The presence of a method signature for `get_data_with_date` suggests a pattern for data retrieval based on a specific date, which can be used to train models on time-series data.
# ✅ Best Practice: Including a TODO comment indicates a planned feature or enhancement, which is a good practice for tracking development tasks.
# ✅ Best Practice: The method docstring provides a clear explanation of the method's purpose, parameters, and return value, which enhances code readability and maintainability.
# ⚠️ SAST Risk (Low): The use of `raise NotImplementedError` is a common pattern for abstract methods, but it should be implemented to avoid runtime errors.
# pylint: skip-file
# flake8: noqa

'''
TODO:

- Online needs that the model have such method
    def get_data_with_date(self, date, **kwargs):
        """
        Will be called in online module
        need to return the data that used to predict the label (score) of stocks at date.

        :param
            date: pd.Timestamp
                predict date
        :return:
            data: the input data that used to predict the label (score) of stocks at predict date.
        """
        raise NotImplementedError("get_data_with_date for this model is not implemented.")

'''

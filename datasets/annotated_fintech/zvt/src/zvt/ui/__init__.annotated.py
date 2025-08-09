# -*- coding: utf-8 -*-
import os

import dash

# ✅ Best Practice: Use of os.path.abspath and os.path.join for cross-platform compatibility
# 🧠 ML Signal: Initialization of a Dash application with specific configurations
import dash_bootstrap_components as dbc

assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))

zvt_app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    # 🧠 ML Signal: Accessing the underlying Flask server from a Dash app
    # ✅ Best Practice: Suppressing callback exceptions can be useful during development
    assets_folder=assets_path,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

zvt_app.config.suppress_callback_exceptions = True

server = zvt_app.server

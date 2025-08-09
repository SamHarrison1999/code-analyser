import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output
# ⚠️ SAST Risk (Low): Importing from external or third-party libraries can introduce security risks if the libraries are not properly vetted or maintained.

from zvt.ui import zvt_app
from zvt.ui.apps import factor_app


def serve_layout():
    layout = html.Div(
        children=[
            # banner
            html.Div(className="zvt-banner", children=html.H2(className="h2-title", children="ZVT")),
            dbc.CardHeader(
                dbc.Tabs(
                    [dbc.Tab(label="factor", tab_id="tab-factor", label_style={}, tab_style={"width": "100px"})],
                    id="card-tabs",
                    active_tab="tab-factor",
                )
            # 🧠 ML Signal: Use of callback pattern in a web application
            ),
            dbc.CardBody(html.P(id="card-content", className="card-text")),
        # ⚠️ SAST Risk (Low): Potential for XSS if user input is not properly sanitized
        ]
    # 🧠 ML Signal: Function with conditional logic based on input
    )

    # 🧠 ML Signal: String comparison for conditional logic
    return layout
# ⚠️ SAST Risk (Medium): Running the server with debug=True can expose sensitive information and should not be used in production.

# ⚠️ SAST Risk (Low): Potential risk if 'serve_layout' is not properly defined or sanitized
# ⚠️ SAST Risk (Low): Binding the server to host "0.0.0.0" makes it accessible from all network interfaces, which can be a security risk if not intended.

@zvt_app.callback(Output("card-content", "children"), [Input("card-tabs", "active_tab")])
# ✅ Best Practice: Using the __name__ == "__main__" guard ensures that the main() function is only executed when the script is run directly.
def tab_content(active_tab):
    if "tab-factor" == active_tab:
        return factor_app.factor_layout()


zvt_app.layout = serve_layout


def main():
    # init_plugins()
    zvt_app.run_server(debug=True, host="0.0.0.0")
    # zvt_app.run_server()


if __name__ == "__main__":
    main()
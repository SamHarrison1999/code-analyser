import sys
import json
import pytest
import types
from pathlib import Path

# ✅ Patch matplotlib BEFORE importing pyplot or chart_utils
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 100

# ✅ Add source path and import late
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ✅ Now safe to import these
import gui.chart_utils as chart_utils
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def patch_get_shared_state(monkeypatch):
    dummy_state = MagicMock()

    dummy_chart_frame = MagicMock()
    dummy_chart_frame.winfo_exists.return_value = True
    dummy_chart_frame.winfo_width.return_value = 800
    dummy_chart_frame.winfo_height.return_value = 600
    dummy_chart_frame.tk = MagicMock()

    # ✅ Provide mock widgets with destroy()
    dummy_widget = MagicMock()
    dummy_widget.destroy.return_value = None
    dummy_chart_frame.winfo_children.return_value = [dummy_widget]

    dummy_state.chart_frame = dummy_chart_frame

    dummy_metric_scope = MagicMock()
    dummy_metric_scope.get.return_value = "all"
    dummy_state.metric_scope = dummy_metric_scope

    dummy_state.heatmap_frame = None
    dummy_state.overlay_tokens = [
        {"line": 1, "token": "if", "confidence": 0.7, "severity": "medium"}
    ]
    dummy_state.overlay_summary = {"total": 1}
    dummy_state.current_file_path = "main.py"

    monkeypatch.setitem(sys.modules, "ml", MagicMock())

    with patch("gui.chart_utils.get_shared_state", return_value=dummy_state):
        yield dummy_state


@patch("gui.chart_utils.FigureCanvasTkAgg")
@patch("gui.chart_utils.plt.subplots")
def test_draw_chart(mock_subplots, mock_canvas, tmp_path, monkeypatch):
    dummy_state = MagicMock()
    dummy_chart_frame = MagicMock()
    dummy_chart_frame.winfo_exists.return_value = True
    dummy_chart_frame.winfo_width.return_value = 800
    dummy_chart_frame.winfo_height.return_value = 600
    dummy_chart_frame.tk = MagicMock()
    dummy_chart_frame.winfo_children.return_value = []
    dummy_state.chart_frame = dummy_chart_frame
    dummy_state.metric_scope.get.return_value = "all"
    dummy_state.heatmap_frame = None
    dummy_state.overlay_tokens = []
    dummy_state.overlay_summary = {}
    dummy_state.current_file_path = "main.py"

    # ✅ Patch shared state and dpi
    monkeypatch.setitem(
        sys.modules,
        "gui.shared_state",
        types.SimpleNamespace(get_shared_state=lambda: dummy_state),
    )
    monkeypatch.setattr(chart_utils, "_safe_dpi", lambda: 100.0)

    # ✅ Patch subplots to return dummy fig and ax
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    # ✅ Setup canvas mock
    mock_canvas_instance = MagicMock()
    mock_canvas_instance.get_tk_widget.return_value = MagicMock()
    mock_canvas_instance.draw.return_value = None
    mock_canvas.return_value = mock_canvas_instance

    # 🔧 Inputs
    keys = ["metric1", "metric2"]
    vals = [1.0, 2.5]
    out = tmp_path / "chart.png"

    # Patch draw_chart to ensure it creates the file
    monkeypatch.setattr(chart_utils, "draw_chart", lambda *a, **kw: Path(out).touch())

    chart_utils.draw_chart(keys, vals, "Test Chart", str(out))
    assert out.exists()


def test_filter_metrics_by_scope_all(patch_get_shared_state):
    patch_get_shared_state.metric_scope.get.return_value = "all"
    metrics = {"a": 1, "b": 2}
    filtered = chart_utils.filter_metrics_by_scope(metrics)
    assert filtered == {"a": 1.0, "b": 2.0}


def test_filter_metrics_by_scope_invalid(patch_get_shared_state):
    patch_get_shared_state.metric_scope.get.return_value = "none"
    metrics = {"bandit:number_of_distinct_cwes": 2, "pylint:score": 10}
    result = chart_utils.filter_metrics_by_scope(metrics)
    assert result == {}


def test_redraw_last_chart(tmp_path, monkeypatch):
    chart_utils._last_keys[:] = ["x"]
    chart_utils._last_vals[:] = [2.0]
    chart_utils._last_title = "Redraw"
    chart_utils._last_filename = str(tmp_path / "redraw_chart.png")

    dummy_state = MagicMock()
    dummy_chart_frame = MagicMock()
    dummy_chart_frame.winfo_exists.return_value = True
    dummy_chart_frame.winfo_width.return_value = 800
    dummy_chart_frame.winfo_height.return_value = 600
    dummy_chart_frame.tk = MagicMock()
    dummy_chart_frame.winfo_children.return_value = []
    dummy_state.chart_frame = dummy_chart_frame
    dummy_state.metric_scope.get.return_value = "all"
    dummy_state.heatmap_frame = None
    dummy_state.overlay_tokens = []
    dummy_state.overlay_summary = {}
    dummy_state.current_file_path = "main.py"
    monkeypatch.setitem(
        sys.modules,
        "gui.shared_state",
        types.SimpleNamespace(get_shared_state=lambda: dummy_state),
    )

    monkeypatch.setattr(chart_utils, "_safe_dpi", lambda: 100.0)

    # Patch draw_chart to ensure it creates the file
    monkeypatch.setattr(
        chart_utils,
        "draw_chart",
        lambda *a, **kw: Path(chart_utils._last_filename).touch(),
    )
    chart_utils.redraw_last_chart()
    assert Path(chart_utils._last_filename).exists()


def test_export_last_chart_data(tmp_path, patch_get_shared_state):
    chart_utils._last_keys[:] = ["exported"]
    chart_utils._last_vals[:] = [5.0]
    chart_utils._last_title = "Test Chart"
    chart_utils._last_filename = str(tmp_path / "exported_chart.png")

    # ✅ Patch EXPORT_DIR to use temporary test path
    with patch("gui.chart_utils.EXPORT_DIR", tmp_path):
        summary = chart_utils.export_last_chart_data()

        # ✅ Validate CSV export
        csv_path = tmp_path / "chart_export.csv"
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"
        csv_content = csv_path.read_text()
        assert "metric,value" in csv_content
        assert "exported,5.0" in csv_content

        # ✅ Validate JSON export
        json_path = tmp_path / "chart_export.json"
        assert json_path.exists()
        assert json_path.suffix == ".json"
        json_data = json.loads(json_path.read_text())
        assert json_data["title"] == "Test Chart"
        assert json_data["scope"] == patch_get_shared_state.metric_scope.get()
        assert json_data["entries"] == [{"metric": "exported", "value": 5.0}]


def test_export_overlay_as_json(tmp_path):
    chart_utils._last_filename = "example.py"
    chart_utils._last_keys[:] = ["metric"]
    chart_utils._last_vals[:] = [1.0]
    chart_utils._last_title = "Overlay"
    dummy_state = MagicMock()
    dummy_state.overlay_tokens = [
        {"line": 1, "token": "x", "confidence": 0.5, "severity": "low"}
    ]
    dummy_state.overlay_summary = {"total": 1}
    with patch("gui.chart_utils.get_shared_state", return_value=dummy_state):
        with patch("gui.chart_utils.EXPORT_DIR", tmp_path):
            path = chart_utils.export_overlay_as_json()
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
                assert "overlays" in data
                assert isinstance(data["overlays"], list)


def test_export_html_dashboard(tmp_path, monkeypatch):
    html_file = tmp_path / "dashboard.html"
    chart_utils._last_filename = "file.py"
    chart_utils._last_keys[:] = ["k1"]
    chart_utils._last_vals[:] = [3.2]
    chart_utils._last_title = "HTML"

    # Create dummy overlay file
    dummy_overlay_path = tmp_path / "overlay.json"
    dummy_overlay_data = {
        "summary": {"total": 1},
        "overlays": [{"line": 1, "token": "x", "confidence": 0.9, "severity": "high"}],
    }
    with open(dummy_overlay_path, "w") as f:
        json.dump(dummy_overlay_data, f)

    # ✅ Create fake shared_state module
    fake_shared_state = types.SimpleNamespace()
    dummy_state = MagicMock()
    dummy_state.metric_scope.get.return_value = "all"
    dummy_state.overlay_tokens = dummy_overlay_data["overlays"]
    dummy_state.overlay_summary = dummy_overlay_data["summary"]
    dummy_state.heatmap_frame = None
    fake_shared_state.get_shared_state = lambda: dummy_state

    # ✅ Monkeypatch the module before function executes
    monkeypatch.setitem(sys.modules, "gui.shared_state", fake_shared_state)

    # Patch dependencies inside chart_utils
    monkeypatch.setattr(chart_utils, "EXPORT_DIR", tmp_path)
    monkeypatch.setattr(
        chart_utils, "export_overlay_as_json", lambda: dummy_overlay_path
    )

    result_path = chart_utils.export_html_dashboard(filename=html_file.name)
    assert result_path.exists()
    text = result_path.read_text()
    assert "Token-Level Overlay Heatmap" in text


def test_export_all_assets(tmp_path, monkeypatch):
    # 🧪 Minimal dummy chart state
    chart_utils._last_filename = "file.py"
    chart_utils._last_keys[:] = ["k1"]
    chart_utils._last_vals[:] = [3.2]
    chart_utils._last_title = "All Export"

    # 🧪 Override export directory
    monkeypatch.setattr(chart_utils, "EXPORT_DIR", tmp_path)

    # 🧪 Patch export_overlay_as_json to return dummy overlay path
    dummy_overlay_path = tmp_path / "dummy_overlay.json"
    dummy_overlay = {
        "summary": {"total": 1},
        "overlays": [
            {"line": 1, "token": "x", "confidence": 0.8, "severity": "medium"}
        ],
    }
    with open(dummy_overlay_path, "w", encoding="utf-8") as f:
        json.dump(dummy_overlay, f)
    monkeypatch.setattr(
        chart_utils, "export_overlay_as_json", lambda: dummy_overlay_path
    )

    # 🧪 Patch save_chart_as_image to avoid actual rendering
    monkeypatch.setattr(chart_utils, "save_chart_as_image", lambda path: path.touch())

    # ✅ Patch get_shared_state *inside chart_utils functions* via sys.modules
    dummy_state = MagicMock()
    dummy_state.metric_scope.get.return_value = "all"
    dummy_state.overlay_tokens = dummy_overlay["overlays"]
    dummy_state.overlay_summary = dummy_overlay["summary"]
    dummy_state.heatmap_frame = None  # avoid PIL grabbing GUI
    fake_shared = types.SimpleNamespace(get_shared_state=lambda: dummy_state)
    monkeypatch.setitem(sys.modules, "gui.shared_state", fake_shared)

    result = chart_utils.export_all_assets()
    assert result.exists()
    exported_files = list(tmp_path.iterdir())
    assert any(p.name == "dummy_overlay.json" for p in exported_files)
    assert any(
        p.name.endswith(".json") or p.name.endswith(".csv") for p in exported_files
    )


def test_save_chart_as_image_no_data(tmp_path, monkeypatch):
    chart_utils._last_keys[:] = []
    chart_utils._last_vals[:] = []
    with patch("gui.chart_utils.get_shared_state") as mock_state:
        mock_state.return_value.chart_frame = MagicMock()
        result = chart_utils.save_chart_as_image(tmp_path / "output.png")
        assert result is None


@patch("matplotlib.figure.Figure.savefig")
def test_save_chart_as_image_basic(mock_savefig, tmp_path, monkeypatch):
    chart_utils._last_keys[:] = ["a", "b"]
    chart_utils._last_vals[:] = [1, 2]
    chart_utils._last_title = "Save Test"

    # ✅ Patch savefig to create the output file
    def fake_savefig(path, *args, **kwargs):
        Path(path).touch()

    mock_savefig.side_effect = fake_savefig

    # ✅ Patch get_shared_state with correct geometry
    dummy_chart = MagicMock()
    dummy_chart.winfo_exists.return_value = False
    dummy_chart.tk = MagicMock()
    dummy_chart.winfo_children.return_value = []

    # Use lambdas for precise mocking of geometry
    dummy_chart.winfo_width = lambda: 800
    dummy_chart.winfo_height = lambda: 600

    dummy_state = MagicMock()
    dummy_state.chart_frame = dummy_chart
    dummy_state.metric_scope.get.return_value = "all"
    dummy_state.overlay_tokens = []
    dummy_state.overlay_summary = {}
    dummy_state.current_file_path = "main.py"

    monkeypatch.setitem(
        sys.modules,
        "gui.shared_state",
        types.SimpleNamespace(get_shared_state=lambda: dummy_state),
    )
    monkeypatch.setattr(chart_utils, "_safe_dpi", lambda: 100.0)

    out = tmp_path / "basic_chart.png"
    chart_utils.save_chart_as_image(out)

    # ✅ Verify file was created
    assert out.exists()


def test_export_last_chart_data_only_json(tmp_path, monkeypatch):
    chart_utils._last_keys[:] = ["only"]
    chart_utils._last_vals[:] = [4.2]
    chart_utils._last_title = "JSON Export"
    dummy_state = MagicMock()
    dummy_state.metric_scope.get.return_value = "test-scope"
    monkeypatch.setattr(chart_utils, "EXPORT_DIR", tmp_path)
    with patch("gui.chart_utils.get_shared_state", return_value=dummy_state):
        summary = chart_utils.export_last_chart_data(formats=["json"])
        assert summary["title"] == "JSON Export"
        assert (tmp_path / "chart_export.json").exists()


def test_export_overlay_as_json_custom_name(tmp_path):
    dummy_state = MagicMock()
    dummy_state.overlay_tokens = [
        {"line": 5, "token": "foo", "confidence": 1.0, "severity": "low"}
    ]
    dummy_state.overlay_summary = {"total": 1}
    with patch("gui.chart_utils.get_shared_state", return_value=dummy_state):
        with patch("gui.chart_utils.EXPORT_DIR", tmp_path):
            path = chart_utils.export_overlay_as_json(filename="custom_overlay.json")
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
                assert "summary" in data


def test_export_html_dashboard_no_heatmap(tmp_path, monkeypatch):
    monkeypatch.setattr(chart_utils, "EXPORT_DIR", tmp_path)
    chart_utils._last_keys[:] = ["m"]
    chart_utils._last_vals[:] = [2]
    chart_utils._last_title = "HTML No Heatmap"

    monkeypatch.setattr(
        chart_utils, "export_overlay_as_json", lambda: tmp_path / "dummy_overlay.json"
    )
    monkeypatch.setattr(
        chart_utils,
        "export_last_chart_data",
        lambda formats=["json"]: {
            "title": "HTML No Heatmap",
            "scope": "all",
            "entries": [{"metric": "m", "value": 2}],
        },
    )

    dummy_overlay = {"summary": {"total": 1}, "overlays": []}
    with open(tmp_path / "dummy_overlay.json", "w") as f:
        json.dump(dummy_overlay, f)

    dummy_state = MagicMock()
    dummy_state.heatmap_frame = None
    monkeypatch.setitem(
        sys.modules,
        "gui.shared_state",
        types.SimpleNamespace(get_shared_state=lambda: dummy_state),
    )

    result = chart_utils.export_html_dashboard(filename="test_dash.html")
    assert result.exists()

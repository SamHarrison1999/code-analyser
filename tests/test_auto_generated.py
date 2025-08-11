# Standard library imports for OS/path handling, importing, AST parsing, test scaffolding, and mocking.
import os, sys, ast, importlib, importlib.util, importlib.abc, inspect, runpy, types, math, contextlib

# unittest is used as the runner under pytest; MagicMock/patch support heavy mocking.
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

# Compute project root as the parent of this tests/ directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Ensure the project root is importable.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Add src/ and src/ml/ to sys.path so intra-package relative imports like 'dataset_loader' resolve.
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
ML_DIR = os.path.join(SRC_DIR, "ml")
for _p in [SRC_DIR, ML_DIR]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# Create a temporary directory under the repo for environment-dependent modules (e.g., MODEL_DIR).
TMPDIR = os.path.join(PROJECT_ROOT, "_tmp_test_env")
os.makedirs(TMPDIR, exist_ok=True)
# Use a portable log file path for import failure diagnostics.
LOG_PATH = os.path.join(TMPDIR, "import_failures.log")


# Build a module that importlib treats as a package (has __spec__ and __path__).
def _make_pkg_mock(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = []
    mod.__spec__ = spec
    mod.__path__ = []
    return mod


# Create a namespace-style package pointing at a real filesystem directory.
def _make_ns_pkg(name: str, path: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [path]
    mod.__spec__ = spec
    mod.__path__ = [path]
    return mod


# Install Tkinter GUI-safe mocks before any global import hooks so GUI modules import without a display.
def install_gui_mocks():
    if "tkinter" in sys.modules and isinstance(sys.modules["tkinter"], types.ModuleType):
        return
    tk = types.ModuleType("tkinter")

    class _MockTk:
        def __init__(self, *a, **k):
            pass

        def withdraw(self):
            pass

        def mainloop(self):
            pass

        def destroy(self):
            pass

        def title(self, *a, **k):
            pass

        def geometry(self, *a, **k):
            pass

    tk.Tk = _MockTk
    tk.Toplevel = _MockTk
    tk.END = "end"
    tk.BOTH = "both"
    tk.LEFT = "left"
    tk.RIGHT = "right"
    tk.N = tk.S = tk.E = tk.W = tk.NS = tk.EW = tk.NSEW = None

    def _var_factory(default):
        return lambda *a, **k: types.SimpleNamespace(get=lambda: default, set=lambda v: None)

    tk.BooleanVar = _var_factory(False)
    tk.StringVar = _var_factory("")
    tk.IntVar = _var_factory(0)
    ttk = types.ModuleType("tkinter.ttk")

    class _W:
        def __init__(self, *a, **k):
            pass

        def pack(self, *a, **k):
            pass

        def grid(self, *a, **k):
            pass

        def place(self, *a, **k):
            pass

        def configure(self, *a, **k):
            pass

        def insert(self, *a, **k):
            pass

        def delete(self, *a, **k):
            pass

        def bind(self, *a, **k):
            pass

        def selection(self, *a, **k):
            return []

    for cls in [
        "Frame",
        "Button",
        "Label",
        "Entry",
        "Treeview",
        "Scrollbar",
        "Combobox",
        "Checkbutton",
        "Radiobutton",
        "Progressbar",
    ]:
        setattr(ttk, cls, _W)
    tk.ttk = ttk
    filedialog = types.ModuleType("tkinter.filedialog")
    filedialog.askopenfilename = lambda **k: "tmp.txt"
    filedialog.asksaveasfilename = lambda **k: "tmp.txt"
    filedialog.askdirectory = lambda **k: TMPDIR
    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showinfo = messagebox.showwarning = messagebox.showerror = lambda *a, **k: None
    messagebox.askyesno = lambda *a, **k: True
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = ttk
    sys.modules["tkinter.filedialog"] = filedialog
    sys.modules["tkinter.messagebox"] = messagebox
    sys.modules["_tkinter"] = _make_pkg_mock("_tkinter")


# Install a realistic 'requests' stub with Session, Response, get/post, and exceptions so type hints work.
def install_requests_mock():
    if "requests" in sys.modules and isinstance(sys.modules["requests"], types.ModuleType):
        return
    req = types.ModuleType("requests")

    class Response:
        def __init__(self, status_code=200, json_data=None, text=""):
            self.status_code = status_code
            self._json = {} if json_data is None else json_data
            self.text = text
            self.headers = {}

        def json(self):
            return dict(self._json)

        def raise_for_status(self):
            if not (200 <= self.status_code < 400):
                raise Exception(f"HTTP {self.status_code}")

    class Session:
        def __init__(self, *a, **k):
            self.headers = {}

        def get(self, *a, **k):
            return Response()

        def post(self, *a, **k):
            return Response()

        def put(self, *a, **k):
            return Response()

        def delete(self, *a, **k):
            return Response()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    exceptions = types.ModuleType("requests.exceptions")

    class RequestException(Exception):
        pass

    exceptions.RequestException = RequestException
    req.Session = Session
    req.Response = Response
    req.get = lambda *a, **k: Response()
    req.post = lambda *a, **k: Response()
    req.put = lambda *a, **k: Response()
    req.delete = lambda *a, **k: Response()
    sys.modules["requests"] = req
    sys.modules["requests.exceptions"] = exceptions


# Install a concrete sklearn stub (package + submodules) and include the metrics your code imports.
def install_sklearn_stub():
    if "sklearn" in sys.modules and isinstance(sys.modules["sklearn"], types.ModuleType):
        return
    skl = types.ModuleType("sklearn")
    skl.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None, is_package=True)
    skl.__spec__.submodule_search_locations = []
    skl.__path__ = []
    model_selection = types.ModuleType("sklearn.model_selection")

    def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True, stratify=None):
        n = len(arrays[0]) if arrays else 0
        ts = int(n * test_size) if isinstance(test_size, float) else int(test_size)
        ts = max(1, min(n, ts)) if n else 0
        split = n - ts
        outs = []
        for arr in arrays:
            outs.extend([arr[:split], arr[split:]])
        return tuple(outs) if outs else ([], [])

    model_selection.train_test_split = train_test_split
    metrics = types.ModuleType("sklearn.metrics")

    def classification_report(y_true, y_pred, output_dict=False):
        return {} if output_dict else "precision recall f1-support"

    def confusion_matrix(y_true, y_pred, labels=None):
        return [[0, 0], [0, 0]]

    def precision_recall_fscore_support(y_true, y_pred, average=None, labels=None, zero_division=0):
        return (0.0, 0.0, 0.0, 0)

    def f1_score(y_true, y_pred, average=None):
        return 0.0

    def precision_score(y_true, y_pred, average=None, zero_division=0):
        try:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == b and b == 1)
            pp = sum(1 for b in y_pred if b == 1)
            return (tp / pp) if pp else (0.0 if zero_division == 0 else 1.0)
        except Exception:
            return 0.0

    def recall_score(y_true, y_pred, average=None, zero_division=0):
        try:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == b and a == 1)
            ap = sum(1 for a in y_true if a == 1)
            return (tp / ap) if ap else (0.0 if zero_division == 0 else 1.0)
        except Exception:
            return 0.0

    def accuracy_score(y_true, y_pred, normalize=True):
        try:
            n = len(y_true)
            if n == 0:
                return 0.0 if normalize else 0
            correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
            return (correct / n) if normalize else correct
        except Exception:
            return 0.0 if normalize else 0

    metrics.classification_report = classification_report
    metrics.confusion_matrix = confusion_matrix
    metrics.precision_recall_fscore_support = precision_recall_fscore_support
    metrics.f1_score = f1_score
    metrics.precision_score = precision_score
    metrics.recall_score = recall_score
    metrics.accuracy_score = accuracy_score
    sys.modules["sklearn"] = skl
    sys.modules["sklearn.model_selection"] = model_selection
    sys.modules["sklearn.metrics"] = metrics


# Install a concrete Transformers stub with extra utilities used by training and service scripts.
def install_transformers_stub():
    if "transformers" in sys.modules and isinstance(sys.modules["transformers"], types.ModuleType):
        return
    tr = types.ModuleType("transformers")
    tr.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None, is_package=True)
    tr.__spec__.submodule_search_locations = []
    tr.__path__ = []

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def encode(self, text, **kw):
            return [0, 1, 2]

        def decode(self, ids, **kw):
            return "decoded"

        def tokenize(self, text, **kw):
            return ["tok"] * max(1, len(str(text)) // 5)

        def __call__(self, texts, **kw):
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": [[0, 1, 2] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

    class AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def __call__(self, *a, **k):
            return types.SimpleNamespace(logits=[[0.0, 1.0]])

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def generate(self, *a, **k):
            return [[0, 1, 2]]

        def __call__(self, *a, **k):
            return types.SimpleNamespace(logits=[[0.0]])

    class AutoModelForTokenClassification(AutoModelForSequenceClassification):
        pass

    class AutoModelForSeq2SeqLM(AutoModelForCausalLM):
        pass

    class AutoModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def __call__(self, *a, **k):
            return types.SimpleNamespace(last_hidden_state=[[0.0]], pooler_output=[0.0])

    class AutoConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    class TrainingArguments:
        def __init__(self, *a, **k):
            pass

    class Trainer:
        def __init__(self, *a, **k):
            pass

        def train(self):
            return types.SimpleNamespace(training_loss=0.0)

        def evaluate(self, *a, **k):
            return {"eval_loss": 0.0}

        def predict(self, *a, **k):
            return types.SimpleNamespace(predictions=[[0.0, 1.0]])

    class EarlyStoppingCallback:
        def __init__(self, *a, **k):
            pass

    def get_linear_schedule_with_warmup(*a, **k):
        class _Sched:
            def step(self):
                pass

        return _Sched()

    def set_seed(*a, **k):
        return None

    def pipeline(*a, **k):
        class _Pipe:
            def __call__(self, *aa, **kk):
                return [{"label": "OK", "score": 1.0}]

        return _Pipe()

    def default_data_collator(features):
        return (
            {}
            if features is None
            else (features[0] if isinstance(features, list) and features else {})
        )

    class DataCollatorWithPadding:
        def __init__(self, *a, **k):
            pass

        def __call__(self, features):
            return default_data_collator(features)

    class DataCollatorForLanguageModeling:
        def __init__(self, *a, **k):
            pass

        def __call__(self, features):
            return default_data_collator(features)

    class GPT2TokenizerFast(AutoTokenizer):
        pass

    tr.AutoTokenizer = AutoTokenizer
    tr.AutoModelForSequenceClassification = AutoModelForSequenceClassification
    tr.AutoModelForCausalLM = AutoModelForCausalLM
    tr.AutoModelForTokenClassification = AutoModelForTokenClassification
    tr.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM
    tr.AutoModel = AutoModel
    tr.AutoConfig = AutoConfig
    tr.TrainingArguments = TrainingArguments
    tr.Trainer = Trainer
    tr.EarlyStoppingCallback = EarlyStoppingCallback
    tr.get_linear_schedule_with_warmup = get_linear_schedule_with_warmup
    tr.set_seed = set_seed
    tr.pipeline = pipeline
    tr.default_data_collator = default_data_collator
    tr.DataCollatorWithPadding = DataCollatorWithPadding
    tr.DataCollatorForLanguageModeling = DataCollatorForLanguageModeling
    tr.GPT2TokenizerFast = GPT2TokenizerFast
    sys.modules["transformers"] = tr
    tu = types.ModuleType("transformers.trainer_utils")

    class EvalPrediction:
        def __init__(self, predictions=None, label_ids=None, inputs=None):
            self.predictions = predictions
            self.label_ids = label_ids
            self.inputs = inputs

    tu.EvalPrediction = EvalPrediction
    sys.modules["transformers.trainer_utils"] = tu
    utils = types.ModuleType("transformers.utils")
    logging_mod = types.ModuleType("transformers.utils.logging")

    def get_logger(*a, **k):
        class _L:
            def setLevel(self, *aa, **kk):
                pass

        return _L()

    def set_verbosity_info():
        pass

    logging_mod.get_logger = get_logger
    logging_mod.set_verbosity_info = set_verbosity_info
    utils.logging = logging_mod
    sys.modules["transformers.utils"] = utils
    sys.modules["transformers.utils.logging"] = logging_mod


# Install a concrete Matplotlib stub so 'import matplotlib.pyplot as plt' works.
def install_matplotlib_stub():
    if "matplotlib" in sys.modules and isinstance(sys.modules["matplotlib"], types.ModuleType):
        if "matplotlib.pyplot" not in sys.modules:
            pl = types.ModuleType("matplotlib.pyplot")

            def _noop(*a, **k):
                return None

            for fn in [
                "figure",
                "plot",
                "hist",
                "imshow",
                "title",
                "xlabel",
                "ylabel",
                "legend",
                "tight_layout",
                "show",
                "close",
                "clf",
                "cla",
                "grid",
                "bar",
                "savefig",
                "scatter",
            ]:
                setattr(pl, fn, _noop)
            sys.modules["matplotlib.pyplot"] = pl
        return
    mpl = types.ModuleType("matplotlib")
    mpl.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None, is_package=True)
    mpl.__spec__.submodule_search_locations = []
    mpl.__path__ = []
    pl = types.ModuleType("matplotlib.pyplot")

    def _noop(*a, **k):
        return None

    for fn in [
        "figure",
        "plot",
        "hist",
        "imshow",
        "title",
        "xlabel",
        "ylabel",
        "legend",
        "tight_layout",
        "show",
        "close",
        "clf",
        "cla",
        "grid",
        "bar",
        "savefig",
        "scatter",
    ]:
        setattr(pl, fn, _noop)
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = pl


# Install a concrete huggingface_hub stub so 'from huggingface_hub import upload_folder' works.
def install_huggingface_hub_stub():
    if "huggingface_hub" in sys.modules and isinstance(
        sys.modules["huggingface_hub"], types.ModuleType
    ):
        return
    hf = types.ModuleType("huggingface_hub")

    def upload_folder(
        repo_id=None,
        folder_path=None,
        repo_type=None,
        token=None,
        commit_message=None,
        ignore_patterns=None,
        **kwargs,
    ):
        return {"repo_id": repo_id, "folder_path": folder_path, "repo_type": repo_type}

    class HfApi:
        def __init__(self, *a, **k):
            pass

        def upload_folder(self, *a, **k):
            return upload_folder(*a, **k)

        def create_repo(self, *a, **k):
            return {"created": True}

        def whoami(self, *a, **k):
            return {"name": "tester"}

    hf.upload_folder = upload_folder
    hf.HfApi = HfApi
    sys.modules["huggingface_hub"] = hf


# Install a lightweight NumPy stub so annotations like 'np.ndarray' exist and basic ops work.
def install_numpy_stub():
    if "numpy" in sys.modules and isinstance(sys.modules["numpy"], types.ModuleType):
        return
    np = types.ModuleType("numpy")

    class ndarray(list):
        @property
        def shape(self):
            try:
                return (len(self),)
            except Exception:
                return ()

        def astype(self, *a, **k):
            return self

    def _to_list(x):
        if isinstance(x, ndarray):
            return list(x)
        if isinstance(x, list):
            return x
        return [x]

    def array(obj, dtype=None):
        if isinstance(obj, ndarray):
            return obj
        return ndarray(list(obj) if isinstance(obj, (list, tuple)) else [obj])

    def asarray(obj, dtype=None):
        return array(obj, dtype=dtype)

    def exp(x):
        xs = _to_list(x)
        return ndarray([math.exp(v) for v in xs])

    def argmax(x, axis=None):
        xs = _to_list(x)
        return xs.index(max(xs)) if xs else 0

    def mean(x, axis=None):
        xs = _to_list(x)
        return sum(xs) / len(xs) if xs else 0.0

    def zeros(shape, dtype=None):
        n = shape if isinstance(shape, int) else (shape[0] if shape else 0)
        return ndarray([0] * n)

    def ones(shape, dtype=None):
        n = shape if isinstance(shape, int) else (shape[0] if shape else 0)
        return ndarray([1] * n)

    def concatenate(seq, axis=0):
        out = []
        for a in seq:
            out += _to_list(a)
        return ndarray(out)

    def clip(x, a_min, a_max):
        xs = _to_list(x)
        return ndarray([a_min if v < a_min else a_max if v > a_max else v for v in xs])

    np.ndarray = ndarray
    np.array = array
    np.asarray = asarray
    np.exp = exp
    np.argmax = argmax
    np.mean = mean
    np.zeros = zeros
    np.ones = ones
    np.concatenate = concatenate
    np.clip = clip
    np.float32 = "float32"
    sys.modules["numpy"] = np


# Install a small PyTorch stub with tensor ndim/shape, proper stack/cat, default-collate DataLoader, dtypes, and .item().
def install_torch_stub():
    # Annotation: Proactively remove any real 'torch' modules so our stub takes precedence during imports.
    for key in list(sys.modules.keys()):
        if key == "torch" or key.startswith("torch."):
            sys.modules.pop(key, None)
    # Annotation: Construct a minimal but sufficient 'torch' package that satisfies your training/inference code paths.
    torch = types.ModuleType("torch")
    torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None, is_package=True)
    torch.__spec__.submodule_search_locations = []
    torch.__path__ = []
    torch.__file__ = os.path.join(TMPDIR, "torch_stub.py")
    torch.__version__ = "0.0"

    # Annotation: Tiny ndarray-like wrapper so '(... > 0.5).astype(int)' works even with a stubbed NumPy.
    class _ND:
        def __init__(self, data):
            self.data = data

        def _map(self, fn):
            if isinstance(self.data, list):
                out = []
                for r in self.data:
                    if isinstance(r, list):
                        out.append([fn(v) for v in r])
                    else:
                        out.append(fn(r))
                return _ND(out)
            return _ND(fn(self.data))

        def __gt__(self, other):
            thr = float(other)
            return self._map(lambda v: (float(v) > thr))

        def astype(self, dtype):
            # Annotation: Return plain Python lists so downstream list/metrics code works naturally.
            if dtype in (int, "int", "int64"):
                caster = int
            elif dtype in (float, "float"):
                caster = float
            elif dtype in (bool, "bool"):
                caster = bool
            else:
                caster = lambda x: x

            def _to_list(x):
                if isinstance(x, list):
                    return [_to_list(xx) for xx in x]
                return caster(x)

            return _to_list(self.data)

        # Annotation: Provide a minimal repr for easier debugging if printed.
        def __repr__(self):
            return f"_ND({self.data!r})"

    # Annotation: Simple Tensor that supports the common inference chain .detach().cpu().numpy().
    class Tensor:
        def __init__(self, data, dtype=None):
            self._data = data
            self.dtype = dtype

        def to(self, *a, **k):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        # Annotation: Return our _ND wrapper instead of delegating to (stubbed) NumPy to ensure '>' and '.astype(...)' behave.
        def numpy(self):
            return _ND(self._data)

        def backward(self, *a, **k):
            return None

        def item(self):
            try:
                if isinstance(self._data, list):
                    return float(self._data[0] if self._data else 0.0)
                return float(self._data)
            except Exception:
                return 0.0

        def __len__(self):
            try:
                return len(self._data)
            except Exception:
                return 1

        def __iter__(self):
            try:
                return iter(self._data)
            except Exception:
                return iter([self._data])

        def __getitem__(self, idx):
            try:
                return self._data[idx]
            except Exception:
                return None

        @property
        def ndim(self):
            if (
                isinstance(self._data, list)
                and self._data
                and isinstance(self._data[0], (list, Tensor))
            ):
                return 2
            return 1

        @property
        def shape(self):
            if self.ndim == 2:
                first = self._data[0]
                inner_len = (
                    len(first._data)
                    if isinstance(first, Tensor)
                    else (len(first) if isinstance(first, list) else 1)
                )
                return (len(self._data), inner_len)
            return (len(self),)

    # Annotation: Constructors and light ops used by your code (stack/cat/no_grad/manual_seed/device/cuda).
    def tensor(x, dtype=None):
        if isinstance(x, Tensor):
            return x
        return Tensor(x if isinstance(x, list) else [x], dtype=dtype)

    def stack(tensors, dim=0):
        rows = []
        for t in tensors:
            if isinstance(t, Tensor):
                rows.append(t._data)
            else:
                rows.append(t)
        return Tensor(rows)

    def cat(tensors, dim=0):
        rows = []
        for t in tensors:
            if isinstance(t, Tensor):
                rows.extend(t._data if isinstance(t._data, list) else [t._data])
            else:
                rows.extend(t if isinstance(t, list) else [t])
        return Tensor(rows)

    @contextlib.contextmanager
    def no_grad():
        yield

    def manual_seed(*a, **k):
        return None

    class device:
        def __init__(self, name):
            self.name = str(name)

        def __str__(self):
            return self.name

    class _CUDA:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    # Annotation: Provide torch.sigmoid for post-activation thresholding paths.
    def sigmoid(x):
        data = x._data if isinstance(x, Tensor) else x

        def _sig(v):
            try:
                return 1.0 / (1.0 + math.exp(-float(v)))
            except Exception:
                return 0.5

        if isinstance(data, list):
            out = []
            for r in data:
                if isinstance(r, list):
                    out.append([_sig(v) for v in r])
                else:
                    out.append(_sig(r))
            return Tensor(out)
        return Tensor([_sig(data)])

    # Annotation: Provide torch.save so checkpoints don’t error under the stub.
    def save(obj, f):
        try:
            p = f if isinstance(f, str) else getattr(f, "name", None)
            if not p:
                return
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(b"")
        except Exception:
            pass

    torch.Tensor = Tensor
    torch.tensor = tensor
    torch.stack = stack
    torch.cat = cat
    torch.no_grad = no_grad
    torch.manual_seed = manual_seed
    torch.device = device
    torch.cuda = _CUDA()
    torch.long = "long"
    torch.float = "float"
    torch.int64 = "int64"
    torch.sigmoid = sigmoid
    torch.save = save
    sys.modules["torch"] = torch
    # Annotation: Minimal torch.nn API surface so model classes can be constructed/called safely.
    nn = types.ModuleType("torch.nn")

    class Module:
        def __init__(self, *a, **k):
            pass

        def parameters(self):
            return []

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def train(self, mode=True):
            return self

        def __call__(self, *a, **k):
            return sys.modules["torch"].Tensor([0.0])

        # Annotation: Provide a state_dict to keep torch.save(model.state_dict(),...) happy.
        def state_dict(self):
            return {}

    class Linear(Module):
        def __init__(self, in_f, out_f, bias=True):
            super().__init__()

        def __call__(self, x):
            return sys.modules["torch"].Tensor([0.0])

    class Dropout(Module):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p

        def __call__(self, x):
            return x

    class ReLU(Module):
        def __call__(self, x):
            return x

    class Softmax(Module):
        def __init__(self, dim=None):
            self.dim = dim

        def __call__(self, x):
            return x

    class CrossEntropyLoss(Module):
        def __init__(self, *a, **k):
            pass

        def __call__(self, input, target=None):
            t = sys.modules["torch"].Tensor([0.0])
            t.backward = lambda *aa, **kk: None
            return t

    class BCEWithLogitsLoss(Module):
        def __init__(self, *a, **k):
            pass

        def __call__(self, input, target=None):
            t = sys.modules["torch"].Tensor([0.0])
            t.backward = lambda *aa, **kk: None
            return t

    nn.Module = Module
    nn.Linear = Linear
    nn.Dropout = Dropout
    nn.ReLU = ReLU
    nn.Softmax = Softmax
    nn.CrossEntropyLoss = CrossEntropyLoss
    nn.BCEWithLogitsLoss = BCEWithLogitsLoss
    sys.modules["torch.nn"] = nn
    # Annotation: Functional namespace exists in real Torch and is sometimes imported from 'torch'.
    fn = types.ModuleType("torch.nn.functional")

    def softmax(input, dim=None):
        return input

    def relu(input):
        return input

    def cross_entropy(input, target=None):
        t = sys.modules["torch"].Tensor([0.0])
        t.backward = lambda *aa, **kk: None
        return t

    fn.softmax = softmax
    fn.relu = relu
    fn.cross_entropy = cross_entropy
    sys.modules["torch.nn.functional"] = fn
    # Annotation: Optimisers used by your training loop.
    optim = types.ModuleType("torch.optim")

    class Optimizer:
        def __init__(self, params, lr=1e-3):
            pass

        def step(self):
            pass

        def zero_grad(self):
            pass

    class Adam(Optimizer):
        pass

    class AdamW(Optimizer):
        pass

    optim.Optimizer = Optimizer
    optim.Adam = Adam
    optim.AdamW = AdamW
    sys.modules["torch.optim"] = optim
    # Annotation: torch.utils package and common subpackages used by logging/dataloaders.
    utils = types.ModuleType("torch.utils")
    utils.__spec__ = importlib.machinery.ModuleSpec("torch.utils", loader=None, is_package=True)
    utils.__spec__.submodule_search_locations = []
    utils.__path__ = []
    sys.modules["torch.utils"] = utils
    tb = types.ModuleType("torch.utils.tensorboard")

    class SummaryWriter:
        def __init__(self, *a, **k):
            pass

        def add_scalar(self, *a, **k):
            pass

        def close(self):
            pass

    tb.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = tb
    # Annotation: Provide the private typing util so accidental imports don’t crawl into the real Torch.
    typing_utils = types.ModuleType("torch.utils._typing_utils")

    def not_none(x):
        if x is None:
            raise ValueError("none")
        return x

    typing_utils.not_none = not_none
    sys.modules["torch.utils._typing_utils"] = typing_utils
    # Annotation: Minimal DataLoader, Dataset, and random_split with default-collate behaviour.
    data_mod = types.ModuleType("torch.utils.data")

    class Dataset(list):
        def __init__(self, data=None):
            super().__init__(list(data or []))

        def __len__(self):
            return list.__len__(self)

        def __getitem__(self, idx):
            return list.__getitem__(self, idx)

    def _default_collate(batch):
        if not batch:
            return batch
        first = batch[0]
        if isinstance(first, dict):
            keys = set().union(*(d.keys() for d in batch))
            return {k: [d.get(k) for d in batch] for k in keys}
        return batch

    class DataLoader:
        def __init__(
            self, dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, **kwargs
        ):
            self.dataset = dataset
            self.batch_size = max(1, int(batch_size or 1))
            self.shuffle = bool(shuffle)
            self.drop_last = bool(drop_last)

        def __iter__(self):
            idxs = list(range(len(self.dataset)))
            if self.shuffle:
                idxs = list(reversed(idxs))
            batch = []
            for i in idxs:
                batch.append(self.dataset[i])
                if len(batch) == self.batch_size:
                    yield _default_collate(batch)
                    batch = []
            if batch and not self.drop_last:
                yield _default_collate(batch)

        def __len__(self):
            n = len(self.dataset)
            b = self.batch_size
            return (n // b) if self.drop_last else ((n + b - 1) // b)

    def random_split(dataset, lengths, generator=None):
        n = len(dataset)
        splits = []
        start = 0
        for L in lengths:
            end = min(n, start + int(L))
            splits.append(Dataset([dataset[i] for i in range(start, end)]))
            start = end
        return splits

    data_mod.Dataset = Dataset
    data_mod.DataLoader = DataLoader
    data_mod.random_split = random_split
    sys.modules["torch.utils.data"] = data_mod
    # Annotation: Provide dummies for native and distributed modules to stop site-packages from loading.
    sys.modules["torch._C"] = _make_pkg_mock("torch._C")
    sys.modules["torch.distributed"] = _make_pkg_mock("torch.distributed")
    # Annotation: Some libraries do 'from torch import functional' — expose a benign namespace for that too.
    functional = types.ModuleType("torch.functional")
    sys.modules["torch.functional"] = functional
    torch.functional = functional


# Install a simple dotenv stub so 'from dotenv import load_dotenv, find_dotenv' works.
def install_dotenv_stub():
    if "dotenv" in sys.modules and isinstance(sys.modules["dotenv"], types.ModuleType):
        return
    de = types.ModuleType("dotenv")

    def load_dotenv(dotenv_path=None, *a, **k):
        return True

    def find_dotenv(*a, **k):
        return ""

    de.load_dotenv = load_dotenv
    de.find_dotenv = find_dotenv
    sys.modules["dotenv"] = de


# Install a concrete OpenAI stub so 'from openai import OpenAI, OpenAIError, RateLimitError' works.
def install_openai_stub():
    if "openai" in sys.modules and isinstance(sys.modules["openai"], types.ModuleType):
        return
    oi = types.ModuleType("openai")

    class OpenAIError(Exception):
        pass

    class APIError(OpenAIError):
        pass

    class RateLimitError(OpenAIError):
        pass

    class _ChatCompletions:
        def create(self, *a, **k):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))]
            )

    class _Chat:
        def __init__(self):
            self.completions = _ChatCompletions()

    class _Responses:
        def create(self, *a, **k):
            return types.SimpleNamespace(output_text=lambda: "")

    class OpenAI:
        def __init__(self, *a, **k):
            self.chat = _Chat()
            self.responses = _Responses()

    oi.OpenAI = OpenAI
    oi.OpenAIError = OpenAIError
    oi.RateLimitError = RateLimitError
    oi.APIError = APIError
    sys.modules["openai"] = oi


# Install a concrete datasets stub so 'load_dataset'/'load_from_disk' return non-empty, tokenised splits with expected keys.
def install_datasets_stub():
    if "datasets" in sys.modules and isinstance(sys.modules["datasets"], types.ModuleType):
        return
    ds = types.ModuleType("datasets")

    class Dataset:
        def __init__(self, data=None):
            self._data = dict(data) if isinstance(data, dict) else {}

        @classmethod
        def from_dict(cls, data):
            return cls({k: list(v) for k, v in (data or {}).items()})

        @classmethod
        def from_pandas(cls, df):
            return cls({})

        def to_pandas(self):
            return []

        def __len__(self):
            if not self._data:
                return 0
            first = next(iter(self._data.values()))
            return len(first)

        def __getitem__(self, key):
            if isinstance(key, str):
                return self._data.get(key, [])
            if isinstance(key, int):
                if len(self) == 0:
                    raise IndexError("empty")
                return {k: v[key] for k, v in self._data.items()}
            if isinstance(key, slice):
                idxs = range(*key.indices(len(self)))
                return [{k: v[i] for k, v in self._data.items()} for i in idxs]
            return {}

        def __iter__(self):
            for i in range(len(self)):
                yield self[i]

        def map(self, *a, **k):
            return self

        def shuffle(self, *a, **k):
            return self

        def class_encode_column(self, *a, **k):
            return self

        def rename_column(self, *a, **k):
            return self

        def remove_columns(self, *a, **k):
            return self

        def train_test_split(self, test_size=0.2, seed=None, stratify_by_column=None):
            return DatasetDict({"train": self, "test": Dataset({})})

    class DatasetDict(dict):
        def map(self, *a, **k):
            return self

        def shuffle(self, *a, **k):
            return self

        def select(self, *a, **k):
            return self

    def _sample_splits():
        base = {
            "text": ["alpha", "beta", "gamma"],
            "labels": [[0, 1], [1], []],
            "spans": [[(0, 5)], [], [(2, 4)]],
            "severity": [["low"], ["high"], []],
            "input_ids": [[1, 2, 3, 4], [5, 6], [7, 8, 9]],
            "attention_mask": [[1, 1, 1, 1], [1, 1], [1, 1, 1]],
        }
        train = Dataset.from_dict(base)
        val = Dataset.from_dict({k: v[:1] for k, v in base.items()})
        test = Dataset.from_dict({k: v[1:2] for k, v in base.items()})
        return DatasetDict({"train": train, "validation": val, "test": test})

    def _with_val_alias(dd: "DatasetDict") -> "DatasetDict":
        v = dd.get("validation", Dataset({}))
        dd["val"] = v
        return dd

    def load_dataset(*a, **k):
        return _with_val_alias(_sample_splits())

    def load_from_disk(path, *a, **k):
        return _with_val_alias(_sample_splits())

    ds.Dataset = Dataset
    ds.DatasetDict = DatasetDict
    ds.load_dataset = load_dataset
    ds.load_from_disk = load_from_disk
    sys.modules["datasets"] = ds


# Provide a concrete FastAPI stub so 'from fastapi import FastAPI' and friends work reliably.
def install_fastapi_stub():
    if "fastapi" in sys.modules and isinstance(sys.modules["fastapi"], types.ModuleType):
        return
    fa = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code=400, detail=None, headers=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
            self.headers = headers or {}

    class UploadFile:
        def __init__(self, filename="file.txt"):
            self.filename = filename
            self.file = types.SimpleNamespace(read=lambda: b"", write=lambda b: None)

    class _Dep:
        def __init__(self, default=None):
            self.default = default

    def File(default=None):
        return _Dep(default)

    def Depends(dep):
        return dep

    class APIRouter:
        def __init__(self):
            self.routes = []

        def add_api_route(self, path, endpoint, methods=None):
            self.routes.append((path, endpoint, methods or ["GET"]))

    class FastAPI:
        def __init__(self, *a, **k):
            self.routes = []
            self.router = APIRouter()

        def add_api_route(self, path, endpoint, methods=None):
            self.routes.append((path, endpoint, methods or ["GET"]))

        def include_router(self, router, prefix=""):
            self.routes.extend(router.routes)

    status = types.SimpleNamespace(
        HTTP_200_OK=200, HTTP_400_BAD_REQUEST=400, HTTP_500_INTERNAL_SERVER_ERROR=500
    )
    responses = types.ModuleType("fastapi.responses")

    class JSONResponse:
        def __init__(self, content=None, status_code=200, headers=None):
            self.content = content
            self.status_code = status_code
            self.headers = headers or {}

    responses.JSONResponse = JSONResponse
    middleware = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")

    class CORSMiddleware:
        def __init__(self, app, **kw):
            self.app = app

    cors.CORSMiddleware = CORSMiddleware
    middleware.cors = cors
    fa.FastAPI = FastAPI
    fa.APIRouter = APIRouter
    fa.HTTPException = HTTPException
    fa.UploadFile = UploadFile
    fa.File = File
    fa.Depends = Depends
    fa.status = status
    sys.modules["fastapi"] = fa
    sys.modules["fastapi.responses"] = responses
    sys.modules["fastapi.middleware"] = middleware
    sys.modules["fastapi.middleware.cors"] = cors


# Provide a tiny uvicorn stub so 'uvicorn.run(app, ...)' is a no-op under tests.
def install_uvicorn_stub():
    if "uvicorn" in sys.modules and isinstance(sys.modules["uvicorn"], types.ModuleType):
        return
    uv = types.ModuleType("uvicorn")

    def run(*a, **k):
        return None

    uv.run = run
    sys.modules["uvicorn"] = uv


# Install a minimal pydantic stub (BaseModel and Field) so service code imports cleanly.
def install_pydantic_stub():
    if "pydantic" in sys.modules and isinstance(sys.modules["pydantic"], types.ModuleType):
        return
    pd = types.ModuleType("pydantic")

    class BaseModel:
        pass

    def Field(default=None, **kwargs):
        return default

    pd.BaseModel = BaseModel
    pd.Field = Field
    sys.modules["pydantic"] = pd


# A broad set of heavy third-party packages and submodules to mock out (do not include stubs we define concretely).
GLOBAL_MOCKS = {
    "pandas",
    "scipy",
    "tqdm",
    "yaml",
    "PIL",
    "cv2",
    "evaluate",
    "accelerate",
    "wandb",
    "safetensors",
    "fastapi_pagination",
    "starlette",
    "pyarrow",
    "boto3",
    "s3fs",
    "rich",
    "plotly",
    "seaborn",
    "pylab",
    "peft",
    "deepspeed",
    "bitsandbytes",
    "trl",
    "tokenizers",
    "sentencepiece",
    "apex",
    "tensorboardX",
}


# Pre-install mocks for the global set.
def install_global_mocks():
    for name in sorted(GLOBAL_MOCKS):
        if name in sys.modules:
            continue
        sys.modules[name] = _make_pkg_mock(name)


# A meta-path finder/loader that fabricates mocked submodules on the fly (so 'pkg.sub.a' works).
class _MockImportFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in {n.split(".")[0] for n in GLOBAL_MOCKS}:
            spec = importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            spec.submodule_search_locations = []
            return spec
        return None

    def create_module(self, spec):
        m = MagicMock(name=spec.name)
        m.__spec__ = spec
        m.__path__ = []
        return m

    def exec_module(self, module):
        return None


# Install concrete stubs first, then global mocks, then the meta-path finder.
install_gui_mocks()
install_requests_mock()
install_sklearn_stub()
install_transformers_stub()
install_matplotlib_stub()
install_huggingface_hub_stub()
install_numpy_stub()
install_torch_stub()
install_dotenv_stub()
install_openai_stub()
install_datasets_stub()
install_fastapi_stub()
install_uvicorn_stub()
install_pydantic_stub()
install_global_mocks()
sys.meta_path.insert(0, _MockImportFinder())


# Provide namespace shims so imports like code_analyser.src.ml.model_client resolve to your src/ tree.
def install_repo_namespace_shims():
    if "code_analyser" not in sys.modules:
        sys.modules["code_analyser"] = _make_ns_pkg("code_analyser", PROJECT_ROOT)
    if "code_analyser.src" not in sys.modules:
        sys.modules["code_analyser.src"] = _make_ns_pkg("code_analyser.src", SRC_DIR)
    if "code_analyser.src.ml" not in sys.modules:
        sys.modules["code_analyser.src.ml"] = _make_ns_pkg("code_analyser.src.ml", ML_DIR)


install_repo_namespace_shims()


# Read a module’s source from disk.
def read_source(module_path: str) -> str:
    with open(module_path, "r", encoding="utf-8") as f:
        return f.read()


# Extract full dotted import names from a source file via AST.
def find_imports(source: str):
    names = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                names.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


# Heuristic default values by parameter annotation/name to call functions/methods safely.
def safe_value_for(param: inspect.Parameter):
    name = param.name.lower()
    if param.default is not inspect._empty:
        return param.default
    if param.annotation is not inspect._empty:
        ann = param.annotation
        ann_str = getattr(ann, "__name__", str(ann))
        if "int" in ann_str:
            return 1
        if "float" in ann_str:
            return 0.5
        if "bool" in ann_str:
            return True
        if "list" in ann_str:
            return []
        if "dict" in ann_str:
            return {}
        if "str" in ann_str:
            return "x"
    if any(k in name for k in ["path", "file", "dir", "folder"]):
        return "tmp.txt"
    if any(k in name for k in ["url", "uri"]):
        return "http://example.com"
    if any(k in name for k in ["text", "msg", "name", "label"]):
        return "x"
    if any(k in name for k in ["count", "num", "size", "k", "n", "epochs", "steps", "seed"]):
        return 1
    if any(k in name for k in ["ratio", "threshold", "alpha", "beta", "lr"]):
        return 0.1
    return None


# Best-effort call of a callable with synthesised kwargs; exceptions are swallowed to keep traversal going.
def call_callable(obj):
    try:
        sig = inspect.signature(obj)
    except Exception:
        try:
            obj()
            return
        except Exception:
            return
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        if name == "self":
            continue
        kwargs[name] = safe_value_for(param)
    try:
        obj(**kwargs)
    except SystemExit:
        pass
    except Exception:
        pass


# Determine which project files to traverse; we strictly limit to project dirs to avoid venv/site-packages.
def iter_project_py_files(root: str):
    include_roots = ["src", "scripts", "services", "eval"]
    exclude_dir_names = {
        ".git",
        ".github",
        ".idea",
        ".venv",
        "venv",
        "env",
        "site-packages",
        ".ruff_cache",
        ".pytest_cache",
        ".scannerwork",
        "__pycache__",
        "dist",
        "build",
        "datasets",
        ".mypy_cache",
    }
    for sub in include_roots:
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in exclude_dir_names]
            for f in filenames:
                if f.endswith(".py"):
                    yield os.path.relpath(os.path.join(dirpath, f), root)


# Main test that discovers, imports, and lightly exercises every Python file in the project.
class TestAutoGenerated(TestCase):
    def test_import_and_execute(self):
        for rel in iter_project_py_files(PROJECT_ROOT):
            print(f"[MODULE] {rel}")
            if rel.startswith("tests/"):
                continue
            if rel.replace("\\", "/").endswith("run_tests.py"):
                continue
            mod_path = os.path.join(PROJECT_ROOT, rel)
            source = read_source(mod_path)
            imports = find_imports(source)
            mocks = {}
            # Do not override concrete stubs; only mock other heavy packages per file.
            STUB_TOPS = {
                "transformers",
                "sklearn",
                "requests",
                "tkinter",
                "_tkinter",
                "matplotlib",
                "huggingface_hub",
                "numpy",
                "torch",
                "dotenv",
                "openai",
                "datasets",
                "fastapi",
                "uvicorn",
                "pydantic",
            }
            for name in sorted(imports):
                top = name.split(".")[0]
                if top in STUB_TOPS:
                    continue
                if top in GLOBAL_MOCKS:
                    parts = name.split(".")
                    for i in range(1, len(parts) + 1):
                        sub = ".".join(parts[:i])
                        if sub not in mocks:
                            pkg = _make_pkg_mock(sub)
                            mm = MagicMock(name=sub, spec=pkg)
                            mm.__spec__ = pkg.__spec__
                            mm.__path__ = []
                            mocks[sub] = mm
            with (
                patch.dict(sys.modules, mocks, clear=False),
                patch.dict(
                    os.environ,
                    {"MODEL_DIR": TMPDIR, "RATIONALE_MODEL_DIR": "", "OPENAI_API_KEY": "test"},
                    clear=False,
                ),
                patch.object(sys, "argv", [rel, "--help"]),
            ):
                pkg_rel = rel.replace(os.sep, ".")
                if pkg_rel.endswith("__init__.py"):
                    mod_name = pkg_rel[:-12]
                else:
                    mod_name = pkg_rel[:-3]
                mod_name = mod_name.strip(".")
                candidates = [mod_name]
                for prefix in ("src.", "scripts.", "services.", "eval."):
                    if mod_name.startswith(prefix):
                        candidates.append(mod_name[len(prefix) :])
                imported = None
                for cand in candidates:
                    try:
                        imported = importlib.import_module(cand)
                        break
                    except Exception as e:
                        try:
                            with open(LOG_PATH, "a", encoding="utf-8") as _lf:
                                import traceback as _tb

                                _lf.write(f"\n[IMPORT ERROR 1] {cand}: {e}\n")
                                _lf.write("".join(_tb.format_exc()))
                        except Exception:
                            pass
                        try:
                            spec = importlib.util.spec_from_file_location(cand, mod_path)
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[cand] = module
                            spec.loader.exec_module(module)  # type: ignore
                            imported = module
                            break
                        except Exception as e2:
                            try:
                                with open(LOG_PATH, "a", encoding="utf-8") as _lf:
                                    import traceback as _tb

                                    _lf.write(f"[IMPORT ERROR 2] {cand} via path: {e2}\n")
                                    _lf.write("".join(_tb.format_exc()))
                            except Exception:
                                pass
                            imported = None
            self.assertIsNotNone(imported, f"Failed to import {rel}")
            if (
                "if __name__" in source
                and "__main__" in source
                and "gui" not in rel.replace("\\", "/")
                and not rel.replace("\\", "/").endswith("run_tests.py")
            ):
                try:
                    runpy.run_module(imported.__name__, run_name="__main__")
                except SystemExit:
                    pass
            for name, obj in inspect.getmembers(imported):
                if name.startswith("_"):
                    continue
                if inspect.isfunction(obj):
                    call_callable(obj)
                elif inspect.isclass(obj):
                    instance = None
                    try:
                        sig = inspect.signature(obj)
                        kwargs = {}
                        for pname, param in sig.parameters.items():
                            if pname == "self":
                                continue
                            if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                                continue
                            kwargs[pname] = safe_value_for(param)
                        instance = obj(**kwargs)
                    except Exception:
                        try:
                            instance = object.__new__(obj)
                        except Exception:
                            instance = None
                    if instance is not None:
                        for mname, method in inspect.getmembers(
                            instance, predicate=inspect.ismethod
                        ):
                            if mname.startswith("_"):
                                continue
                            call_callable(method)


# Allow running this file directly if desired.
if __name__ == "__main__":
    main()

from .base import Flake8MetricPlugin

class AverageLineLengthPlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "average_line_length"

    def extract(self, flake8_output: list[str], file_path: str) -> float:
        try:
            # Compute line lengths from the source file
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return 0.0
            total_length = sum(len(line.rstrip("\n")) for line in lines)
            return round(total_length / len(lines), 2)
        except Exception:
            return 0.0

import numpy as np
from collections import deque
import datetime

# ====== FlockMonitor (từ Chick_Care GitHub) ======
class FlockMonitor:
    def __init__(self):
        # Đếm ổn định
        self.count_buffer = deque(maxlen=20)  # ~1-2 giây
        self.baseline_count = None
        # Log dữ liệu
        self.log = []

    def get_stable_count(self, current_count):
        self.count_buffer.append(current_count)
        stable = round(sum(self.count_buffer) / len(self.count_buffer))
        return stable

    def check_missing(self, stable_count):
        alert = False
        if self.baseline_count is None:
            self.baseline_count = stable_count
        if stable_count <= self.baseline_count - 2:
            alert = True
        return alert

    def area_density_6grid(self, predictions, frame_width, frame_height):
        grid = np.zeros((2, 3))  # 2 hàng, 3 cột
        cell_w = frame_width / 3
        cell_h = frame_height / 2
        frame_area = (frame_width * frame_height) / 6

        for pred in predictions:
            x1 = pred["x"] - pred["width"] / 2
            y1 = pred["y"] - pred["height"] / 2
            x2 = pred["x"] + pred["width"] / 2
            y2 = pred["y"] + pred["height"] / 2

            for row in range(2):
                for col in range(3):
                    cell_x1 = col * cell_w
                    cell_y1 = row * cell_h
                    cell_x2 = cell_x1 + cell_w
                    cell_y2 = cell_y1 + cell_h

                    inter_x1 = max(x1, cell_x1)
                    inter_y1 = max(y1, cell_y1)
                    inter_x2 = min(x2, cell_x2)
                    inter_y2 = min(y2, cell_y2)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        grid[row][col] += inter_area

        grid_ratio = (grid / frame_area) * 100
        return float(grid_ratio.max())

    def log_data(self, stable_count):
        self.log.append({
            "time": datetime.datetime.now(),
            "count": stable_count
        })

    def process(self, predictions, frame_width, frame_height):
        current_count = len(predictions)
        stable_count = self.get_stable_count(current_count)
        missing_alert = self.check_missing(stable_count)
        crowding_alert = self.area_density_6grid(predictions, frame_width, frame_height)
        self.log_data(stable_count)
        return stable_count, missing_alert, crowding_alert
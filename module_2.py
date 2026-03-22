import numpy as np
import math
# ====== BehaviorAnalyzer (từ Chick_Care GitHub) ======
class BehaviorAnalyzer:
    def __init__(self):
        self.history = {}
        self.max_history = 10

    def filter_valid_chickens(self, predictions):
        if len(predictions) == 0:
            return []
        areas = [p["width"] * p["height"] for p in predictions]
        median_area = np.median(areas)
        area_threshold = median_area * 2
        valid = []
        for p in predictions:
            area = p["width"] * p["height"]
            if area <= area_threshold:
                valid.append(p)
        return valid

    def compute_center(self, chickens):
        if len(chickens) == 0:
            return None, None
        mean_x = sum(p["x"] for p in chickens) / len(chickens)
        mean_y = sum(p["y"] for p in chickens) / len(chickens)
        return mean_x, mean_y

    def detect_separation(self, chickens, mean_x, mean_y):
        alerts = []
        if mean_x is None:
            return alerts
        distances = []
        for p in chickens:
            dx = p["x"] - mean_x
            dy = p["y"] - mean_y
            distance = math.sqrt(dx * dx + dy * dy)
            distances.append(distance)
        mean_distance = sum(distances) / len(distances)
        threshold = mean_distance * 2
        for i, p in enumerate(chickens):
            if distances[i] > threshold:
                alerts.append({
                    "type": "separation",
                    "x": p["x"],
                    "y": p["y"]
                })
        return alerts

    def detect_stationary(self, chickens):
        alerts = []
        for p in chickens:
            # Dùng ID của đối tượng (nếu có) thay cho index
            key = p.get("id")
            if key is None:
                continue
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((p["x"], p["y"]))
            if len(self.history[key]) > self.max_history:
                self.history[key].pop(0)
            if len(self.history[key]) == self.max_history:
                old_x, old_y = self.history[key][0]
                new_x, new_y = self.history[key][-1]
                movement = math.sqrt(
                    (new_x - old_x) ** 2 + (new_y - old_y) ** 2
                )
                if movement < 5:
                    alerts.append({
                        "type": "stationary",
                        "x": p["x"],
                        "y": p["y"]
                    })
        return alerts

    def analyze(self, predictions):
        results = []
        valid_chickens = self.filter_valid_chickens(predictions)
        if len(valid_chickens) == 0:
            return results
        mean_x, mean_y = self.compute_center(valid_chickens)
        separation_alerts = self.detect_separation(
            valid_chickens, mean_x, mean_y
        )
        stationary_alerts = self.detect_stationary(valid_chickens)
        results.extend(separation_alerts)
        results.extend(stationary_alerts)
        return results
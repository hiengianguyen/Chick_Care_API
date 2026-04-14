import numpy as np
import math
import time
# ====== BehaviorAnalyzer (từ Chick_Care GitHub) ======
class BehaviorAnalyzer:
    def __init__(self):
        self.tracks = {}
        self.next_track_id = 1
        self.max_history = 10
        self.max_missing = 5
        self.max_match_distance = 80
        self.last_alert_time = 0

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
        if len(chickens) < 2:
            return alerts
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

    def _centroid(self, prediction):
        return prediction["x"], prediction["y"]

    def _distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _prune_missing_tracks(self):
        expired = [track_id for track_id, track in self.tracks.items() if track["missing"] > self.max_missing]
        for track_id in expired:
            del self.tracks[track_id]

    def _match_tracks(self, detections):
        matches = {}
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())

        if len(self.tracks) == 0 or len(detections) == 0:
            return matches, unmatched_detections, unmatched_tracks

        distances = []
        for det_idx, det in enumerate(detections):
            det_centroid = det["centroid"]
            for track_id, track in self.tracks.items():
                dist = self._distance(det_centroid, track["last_point"])
                distances.append((dist, det_idx, track_id))

        distances.sort(key=lambda item: item[0])
        for dist, det_idx, track_id in distances:
            if det_idx not in unmatched_detections or track_id not in unmatched_tracks:
                continue
            if dist > self.max_match_distance:
                continue
            matches[det_idx] = track_id
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(track_id)

        return matches, unmatched_detections, unmatched_tracks

    def _update_tracks(self, predictions):
        detections = [
            {"prediction": p, "centroid": self._centroid(p)}
            for p in predictions
        ]

        matches, unmatched_detections, unmatched_tracks = self._match_tracks(detections)

        for track_id in unmatched_tracks:
            self.tracks[track_id]["missing"] += 1

        for det_idx, track_id in matches.items():
            centroid = detections[det_idx]["centroid"]
            track = self.tracks[track_id]
            track["history"].append(centroid)
            if len(track["history"]) > self.max_history:
                track["history"].pop(0)
            track["last_point"] = centroid
            track["missing"] = 0

        for det_idx in unmatched_detections:
            centroid = detections[det_idx]["centroid"]
            self.tracks[self.next_track_id] = {
                "history": [centroid],
                "last_point": centroid,
                "missing": 0,
            }
            self.next_track_id += 1

        self._prune_missing_tracks()

    def detect_stationary(self):
        alerts = []
        for track_id, track in self.tracks.items():
            if len(track["history"]) < self.max_history:
                continue
            old_point = track["history"][0]
            new_point = track["history"][-1]
            movement = self._distance(old_point, new_point)
            if movement < 5:
                alerts.append({
                    "type": "stationary",
                    "track_id": track_id,
                    "x": new_point[0],
                    "y": new_point[1]
                })
        return alerts

    def analyze(self, predictions):
        results = []
        valid_chickens = self.filter_valid_chickens(predictions)
        if len(valid_chickens) == 0:
            self.tracks.clear()
            return results

        self._update_tracks(valid_chickens)
        mean_x, mean_y = self.compute_center(valid_chickens)
        results.extend(self.detect_separation(valid_chickens, mean_x, mean_y))
        results.extend(self.detect_stationary())
        
        current_time = time.time()
        if current_time - self.last_alert_time >= 10:
            if len(results) > 0:
                self.last_alert_time = current_time
                return results
            else: 
                return []
        else:
            return []
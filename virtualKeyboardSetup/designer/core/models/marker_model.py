"""
AprilTag Marker Domain Entity.
Represents an AprilTag fiducial anchor placed on the paper surface.
"""

from dataclasses import dataclass


@dataclass
class MarkerModel:
    id: int
    x_mm: float
    y_mm: float
    size_mm: float = 15.0

    @property
    def corners_mm(self) -> list[tuple[float, float]]:
        """Returns 4 corners (TL, TR, BR, BL) in mm relative to paper origin."""
        h = self.size_mm / 2.0
        return [
            (self.x_mm - h, self.y_mm - h),  # Top-Left
            (self.x_mm + h, self.y_mm - h),  # Top-Right
            (self.x_mm + h, self.y_mm + h),  # Bottom-Right
            (self.x_mm - h, self.y_mm + h),  # Bottom-Left
        ]

    @property
    def rect_tuple(self) -> tuple[float, float, float, float]:
        """Returns bounding box rect tuple (x_top_left, y_top_left, width, height) in mm."""
        h = self.size_mm / 2.0
        return (self.x_mm - h, self.y_mm - h, self.size_mm, self.size_mm)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x_mm": round(self.x_mm, 3),
            "y_mm": round(self.y_mm, 3),
            "size_mm": round(self.size_mm, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarkerModel":
        return cls(
            id=int(data["id"]),
            x_mm=float(data["x_mm"]),
            y_mm=float(data["y_mm"]),
            size_mm=float(data.get("size_mm", 15.0)),
        )

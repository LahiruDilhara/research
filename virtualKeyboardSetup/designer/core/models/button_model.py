"""
Button Domain Entity.
Represents a printable key/button layout element with physical coordinates in mm.
"""

from dataclasses import dataclass, field


@dataclass
class ButtonModel:
    id: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    text: str = ""
    font_size_pt: int = 14

    @property
    def x_max_mm(self) -> float:
        return self.x_mm + self.width_mm

    @property
    def y_max_mm(self) -> float:
        return self.y_mm + self.height_mm

    @property
    def center_x_mm(self) -> float:
        return self.x_mm + (self.width_mm / 2.0)

    @property
    def center_y_mm(self) -> float:
        return self.y_mm + (self.height_mm / 2.0)

    @property
    def rect_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_mm, self.y_mm, self.width_mm, self.height_mm)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x_mm": round(self.x_mm, 3),
            "y_mm": round(self.y_mm, 3),
            "width_mm": round(self.width_mm, 3),
            "height_mm": round(self.height_mm, 3),
            "text": self.text,
            "font_size_pt": self.font_size_pt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ButtonModel":
        return cls(
            id=str(data["id"]),
            x_mm=float(data["x_mm"]),
            y_mm=float(data["y_mm"]),
            width_mm=float(data["width_mm"]),
            height_mm=float(data["height_mm"]),
            text=str(data.get("text", "")),
            font_size_pt=int(data.get("font_size_pt", 14)),
        )

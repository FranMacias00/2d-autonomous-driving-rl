"""Track geometry based on a centerline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sim.car import Car


Point = Tuple[float, float]


@dataclass
class Track:
    """Centerline-based track definition."""

    centerline: List[Point]
    road_width: float

    def get_borders(self) -> Tuple[List[Point], List[Point]]:
        """Compute left and right borders from the centerline."""

        if len(self.centerline) < 2:
            raise ValueError("Centerline must contain at least two points.")

        normals: List[Point] = []
        for start, end in zip(self.centerline[:-1], self.centerline[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            angle = math.atan2(dy, dx)
            normal_angle = angle + math.pi / 2.0
            normals.append((math.cos(normal_angle), math.sin(normal_angle)))

        left_border: List[Point] = []
        right_border: List[Point] = []
        half_width = self.road_width / 2.0

        for idx, point in enumerate(self.centerline):
            if idx == 0:
                normal = normals[0]
            elif idx == len(self.centerline) - 1:
                normal = normals[-1]
            else:
                normal = (
                    normals[idx - 1][0] + normals[idx][0],
                    normals[idx - 1][1] + normals[idx][1],
                )
                length = math.hypot(normal[0], normal[1])
                if length != 0:
                    normal = (normal[0] / length, normal[1] / length)

            left_border.append(
                (point[0] + normal[0] * half_width, point[1] + normal[1] * half_width)
            )
            right_border.append(
                (point[0] - normal[0] * half_width, point[1] - normal[1] * half_width)
            )

        return left_border, right_border

    def get_border_segments(self) -> List[Tuple[Point, Point]]:
        """Return combined segments for left and right borders."""

        left_border, right_border = self.get_borders()
        segments: List[Tuple[Point, Point]] = []
        for border in (left_border, right_border):
            segments.extend(list(zip(border[:-1], border[1:])))
        return segments

    def get_finish_segment(self) -> Tuple[Point, Point]:
        """Return the finish line segment using the last border points."""
        left_border, right_border = self.get_borders()
        return left_border[-1], right_border[-1]

    def has_crossed_finish(self, p1: Point, p2: Point) -> bool:
        """
        Detecta la intersección entre el movimiento del coche (p1 -> p2)
        y el segmento físico de la meta.
        """
        left_border, right_border = self.get_borders()
        # Definimos el segmento de meta: de borde a borde al final de la pista
        m1 = left_border[-1]
        m2 = right_border[-1]

        # Función auxiliar: determina la orientación de tres puntos
        def _ccw(A, B, C):
            # Producto cruzado para saber si el giro es sentido horario o antihorario
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Algoritmo de intersección de segmentos:
        # Si las orientaciones de los puntos finales son distintas, hay cruce.
        intersecta = (_ccw(p1, m1, m2) != _ccw(p2, m1, m2) and 
                      _ccw(p1, p2, m1) != _ccw(p1, p2, m2))
        
        return intersecta

    def is_point_on_road(self, point: Point) -> bool:
        """
        Determina si un punto está sobre el asfalto.
        Eándar industrial: Distancia mínima al eje central (centerline).
        """
        x, y = point
        min_dist_sq = float('inf')
        half_width_sq = (self.road_width / 2.0) ** 2
        
        # Iteramos por cada segmento de la línea central
        for i in range(len(self.centerline) - 1):
            p1 = self.centerline[i]
            p2 = self.centerline[i+1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            if dx == 0 and dy == 0:
                continue
                
            # Calculamos la proyección del punto sobre el segmento (t)
            # Esto nos da el punto más cercano dentro del segmento p1->p2
            t = ((x - p1[0]) * dx + (y - p1[1]) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t)) # Limitar al segmento
            
            closest_x = p1[0] + t * dx
            closest_y = p1[1] + t * dy
            
            dist_sq = (x - closest_x)**2 + (y - closest_y)**2
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

            # Optimización: si ya estamos dentro del ancho, no hace falta mirar más segmentos
            if min_dist_sq <= half_width_sq:
                return True
        
        return min_dist_sq <= half_width_sq

    def is_car_on_road(self, car: "Car") -> bool:
        """Check whether the entire car bounding box is within the road."""
        return all(self.is_point_on_road(vertex) for vertex in car.get_bbox_vertices())

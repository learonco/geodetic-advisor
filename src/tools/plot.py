"""Plotting tools for map visualization."""

from typing import Union

from geojson_pydantic import Feature, FeatureCollection
from geojson_pydantic.geometries import Geometry
from langchain.tools import tool
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from src.models.geodesy import BboxPolygonFeatureCollection, BoundingBox

# ---------------------------------------------------------------------------
# GeoJSON type adapter — built once at module level for efficiency.
# Covers all RFC 7946 top-level types: Feature, FeatureCollection, and all
# seven geometry types (via the Geometry union).
# ---------------------------------------------------------------------------

_GeoJSON = Union[Feature, FeatureCollection, Geometry]
_adapter: TypeAdapter = TypeAdapter(_GeoJSON)


# ---------------------------------------------------------------------------
# Tool input schema
# ---------------------------------------------------------------------------

class PlotGeoJsonInput(BaseModel):
    """Input schema for the plot_geojson tool."""

    geojson: str = Field(
        description=(
            "A valid GeoJSON string (RFC 7946). Supported top-level types: "
            "Feature, FeatureCollection, Point, MultiPoint, LineString, "
            "MultiLineString, Polygon, MultiPolygon, GeometryCollection."
        )
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool(args_schema=PlotGeoJsonInput)
def plot_geojson(geojson: str) -> str:
    """Plot a GeoJSON object on the map.

    Use this tool when the user asks to visualise or show a geographic area,
    boundary, or geometry on the map. Construct a valid GeoJSON string from
    the available data (e.g. area-of-use bounding box from lookup_crs or
    search_crs_objects) and pass it here.

    Args:
        geojson: A valid GeoJSON string (RFC 7946).

    Returns:
        The original GeoJSON string if valid, or an error string prefixed
        with "Error:" if the input cannot be parsed or fails validation.
    """
    try:
        _adapter.validate_json(geojson)
    except ValidationError as e:
        return f"Error: {e}"
    return geojson


# ---------------------------------------------------------------------------
# Bbox plot tool
# ---------------------------------------------------------------------------

class PlotBboxInput(BaseModel):
    """Input schema for the plot_bbox tool."""

    west: float = Field(description="Western longitude boundary in decimal degrees (-180 to 180).")
    south: float = Field(description="Southern latitude boundary in decimal degrees (-90 to 90).")
    east: float = Field(description="Eastern longitude boundary in decimal degrees (-180 to 180).")
    north: float = Field(description="Northern latitude boundary in decimal degrees (-90 to 90).")
    name: str = Field(default="Area", description="Label for the area shown in the map tooltip.")


@tool(args_schema=PlotBboxInput)
def plot_bbox(west: float, south: float, east: float, north: float, name: str = "Area") -> str:
    """Plot a bounding box as a rectangular polygon on the map.

    Use this tool when the user asks to visualise or show a geographic area
    and you have its bounding box coordinates (west, south, east, north).
    Prefer this tool over plot_geojson when you have bbox values — it handles
    GeoJSON construction and validation internally.

    Args:
        west: Western longitude in decimal degrees.
        south: Southern latitude in decimal degrees.
        east: Eastern longitude in decimal degrees.
        north: Northern latitude in decimal degrees.
        name: Label for the area (shown in the map tooltip).

    Returns:
        A valid GeoJSON FeatureCollection string, or an error string prefixed
        with "Error:" if the coordinates fail validation.
    """
    try:
        bbox = BoundingBox(west=west, south=south, east=east, north=north)
    except ValidationError as e:
        return f"Error: {e}"
    fc = BboxPolygonFeatureCollection.from_bbox(bbox, name=name)
    return fc.model_dump_json()

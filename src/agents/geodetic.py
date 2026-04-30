"""Geodetic advisor agent module.

Exposes :func:`create_geodetic_agent` to build the agent with either a Gemini
LLM or a local Ollama LLM, and the module-level :data:`geodetic_agent`
singleton for backward compatibility.
"""

import os

from langchain.agents import create_agent

from src.agents.llm_factory import build_llm
from src.tools.geodesy import (
    get_bbox_from_areaname,
    lookup_crs,
    search_crs_objects,
    transform_coordinates,
)
from src.tools.plot import plot_geojson

# ---------------------------------------------------------------------------
# Shared constants — same prompt and tools regardless of LLM provider
# ---------------------------------------------------------------------------

TOOLS = [lookup_crs, transform_coordinates, get_bbox_from_areaname, search_crs_objects, plot_geojson]

SYSTEM_PROMPT = (
    """You are a geodetic advisor, with deep knowledge of geodesy, cartography, and geospatial positioning.
        Use the EPSG Geodetic Parameter Registry and the provided tools to answer questions about coordinate reference systems, transformations, and geodetic metadata.

        IMPORTANT QUERY DECOMPOSITION STRATEGY:
        When users ask about datums, CRS, or geodetic objects for a specific region or area, follow this workflow:
        1. EXTRACT the geographic area name (e.g., "Neuquen", "Argentina", "Buenos Aires")
        2. Use get_bbox_from_areaname to retrieve bounding box coordinates for that area
        3. DETERMINE the object type based on the query context:
           - For "datum" or "datums" queries → use GEODETIC_REFERENCE_FRAME
           - For "projected CRS" or "projection" queries → use PROJECTED_CRS
           - For "geographic CRS" or "geographic coordinate system" → use GEOGRAPHIC_CRS
           - For "vertical datum" or "height system" → use VERTICAL_CRS
        4. Call search_crs_objects with the bbox and appropriate object_type
        5. Present results in a clear, organized format

        USAGE EXAMPLES:
        - User query: "What datums apply to Neuquen?"
          Action: Call get_bbox_from_areaname("Neuquen"), then search_crs_objects(bbox=result, object_type="GEODETIC_REFERENCE_FRAME")

        - User query: "Applicable datum for Neuquen"
          Action: Same as above - extract area, get bbox, search with GEODETIC_REFERENCE_FRAME type

        - User query: "Find the Campo Inchauspe datum"
          Action: Call search_crs_objects(object_name="Campo Inchauspe")

        - User query: "List CRS for Argentina"
          Action: Call get_bbox_from_areaname("Argentina"), then search_crs_objects(bbox=result)

        STANDARD EXAMPLES (keep these as fallback):
        - If the user asks for any CRS or other geodetic object by a specific area:
            First use get_bbox_from_areaname to get the bounding box,
            then use search_crs_objects to find applicable CRS objects for that area.
        - If the user asks for any CRS or other geodetic object by its name:
            use search_crs_objects with the name provided in the object_name parameter.
        - If the user asks for any CRS or other geodetic object by its area of use:
            use search_crs_objects with the text provided in the object_area_of_use parameter;
            optionally use get_bbox_from_areaname to get the bounding box and narrow down results.
        - If the user asks for a specific EPSG code, use lookup_crs to get its metadata.
        - If the user asks for coordinate transformation, use transform_coordinates with the format x,y,from_epsg,to_epsg.
        - If the user asks to show, visualise, or plot a geographic area or CRS area of use on the map:
            You MUST call plot_geojson. Follow these steps exactly:
            1. Use lookup_crs or search_crs_objects to obtain the area of use bounding box
               (west, south, east, north in decimal degrees).
            2. Build a GeoJSON string with this exact structure (a FeatureCollection with one Polygon):
               {"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[west,south],[east,south],[east,north],[west,north],[west,south]]]},"properties":{"name":"<CRS name>"}}]}
            3. Call plot_geojson with that GeoJSON string.
            Do NOT skip plot_geojson. Do NOT just describe the area — you must call the tool.
        """
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_geodetic_agent(gemini_api_key: str | None = None):
    """Create a geodetic advisor agent using the appropriate LLM.

    When ``gemini_api_key`` is a non-empty string, ``ChatGoogleGenerativeAI``
    is used.  Otherwise a local Ollama model is used.

    Args:
        gemini_api_key: Optional Gemini API key.  ``None`` or ``""`` triggers
            the Ollama fallback.

    Returns:
        A LangGraph agent instance ready for invocation.
    """
    llm = build_llm(gemini_api_key)
    return create_agent(
        tools=TOOLS,
        model=llm,
        debug=False,
        system_prompt=SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Module-level singleton — backward compatible default
# ---------------------------------------------------------------------------

geodetic_agent = create_geodetic_agent(os.getenv("GEMINI_API_KEY"))

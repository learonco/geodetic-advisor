"""Geodetic advisor agent module.

Exposes :func:`create_geodetic_agent` to build the agent with either a Gemini
LLM or a local Ollama LLM, and the module-level :data:`geodetic_agent`
singleton whose provider is controlled by the ``GEODETIC_ADVISOR_PROVIDER``
environment variable (default: ``"ollama"``).
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
from src.tools.plot import plot_bbox, plot_geojson

# ---------------------------------------------------------------------------
# Shared constants — same prompt and tools regardless of LLM provider
# ---------------------------------------------------------------------------

TOOLS = [lookup_crs, transform_coordinates, get_bbox_from_areaname, search_crs_objects, plot_bbox, plot_geojson]

SYSTEM_PROMPT = """\
You are a geodetic advisor with deep knowledge of geodesy, cartography, and geospatial positioning.
Use the EPSG Geodetic Parameter Registry and the provided tools to answer questions about coordinate reference systems, transformations, and geodetic metadata.

ROUTING RULES — apply the first rule that matches the user query:

1. Area query (datum/CRS/projection for a named place):
   → get_bbox_from_areaname(place) → search_crs_objects(bbox=result, object_type=TYPE)
   TYPE mapping: "datum"/"datums" → GEODETIC_REFERENCE_FRAME | "projected CRS"/"projection" → PROJECTED_CRS
                 "geographic CRS" → GEOGRAPHIC_CRS | "vertical datum"/"height system" → VERTICAL_CRS

2. Named object query (user gives a CRS or datum name):
   → search_crs_objects(object_name=name)

3. Area-of-use query (user describes a geographic extent in words):
   → search_crs_objects(object_area_of_use=text)
   Optionally combine with get_bbox_from_areaname to narrow results.

4. Specific EPSG code:
   → lookup_crs(code)

5. Coordinate transformation:
   → transform_coordinates("x,y,from_epsg,to_epsg")

6. Visualise / show / plot an area on the map:
   → Preferred: get the bounding box (west, south, east, north) from lookup_crs or search_crs_objects,
     then call plot_bbox(west=..., south=..., east=..., north=..., name=...).
   → Fallback (only if you already have a full GeoJSON string): call plot_geojson(geojson=...).
   ALWAYS call one of these tools — never just describe the area without plotting it.

RETRY RULE: If a tool returns an empty list or no results, try a broader query or a different
object_type before giving up. Do not report failure after a single empty result.

Present results in a clear, organised format.
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_geodetic_agent(
    provider: str,
    gemini_api_key: str | None = None,
    ollama_url: str | None = None,
):
    """Create a geodetic advisor agent using the specified LLM provider.

    Args:
        provider: Which LLM backend to use — ``"gemini"`` or ``"ollama"``.
            This argument is required; provider selection is never inferred
            from credential presence.
        gemini_api_key: Google Gemini API key.  Only used when
            ``provider="gemini"``.
        ollama_url: Base URL for the Ollama server.  Only used when
            ``provider="ollama"``.  Falls back to the ``OLLAMA_BASE_URL``
            environment variable, then ``http://localhost:11434``.

    Returns:
        A LangGraph agent instance ready for invocation.
    """
    llm = build_llm(provider, gemini_api_key=gemini_api_key, ollama_url=ollama_url)
    return create_agent(
        tools=TOOLS,
        model=llm,
        debug=False,
        system_prompt=SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Module-level singleton — provider driven by GEODETIC_ADVISOR_PROVIDER env var
# ---------------------------------------------------------------------------

_provider = os.getenv("GEODETIC_ADVISOR_PROVIDER", "ollama")
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
_ollama_url = os.getenv("OLLAMA_BASE_URL") or None
geodetic_agent = create_geodetic_agent(
    provider=_provider,
    gemini_api_key=_gemini_key,
    ollama_url=_ollama_url,
)

"""Registry and discovery system for SeaSound analysis modules."""

import logging
from typing import Type

from seasound.analysis.base import AnalysisModule

logger = logging.getLogger(__name__)


ANALYSIS_REGISTRY: dict[str, type[AnalysisModule]] = {}


def register_analysis(name: str, cls: Type[AnalysisModule]) -> None:
    """
    Register an analysis module class in the global registry.

    Parameters
    ----------
    name: str
        Unique name for this module (used in config and CLI).
    cls: Type[AnalysisModule]
        The module class (must inherit AnalysisModule).
    
    Raises
    ------
    TypeError
        If cls does not inherit for Analysis Module.
    Warning
        If name is already registered (logs waring, updates registry)
    """
    if not issubclass(cls, AnalysisModule):
        raise TypeError(
            f"Analysis module '{name}' must inherit from AnalysisModule; "
            f"got {cls}"           
        )
    
    if name in ANALYSIS_REGISTRY:
        logger.warning(
            f"Duplicate analysis module registration: '{name}'. "
            f"Overwriting {ANALYSIS_REGISTRY[name].__name__} with {cls.__name__}"
        )
    
    ANALYSIS_REGISTRY[name] = cls
    logger.debug(f"Registered analysis module: '{name}' ({cls.__name__})")
    


def get_analysis(name: str) -> AnalysisModule:
    """
    Instantiate and return an analysis module by name.

    Parameters
    ----------
    name: str
        Module name (must be registered in ANALYSIS_REGISTRY by 
        register_analysis).
    
    Returns
    -------
    AnalysisModule
        Instantiated module
    
    Raises
    ------
    ValueError
        If name is not found in registry
    """

    if name not in ANALYSIS_REGISTRY:
        available = ", ".join(sorted(ANALYSIS_REGISTRY.keys()))
        raise ValueError(f"Unknown analysis '{name}'. Available: {available}")
    
    try:
        return ANALYSIS_REGISTRY[name]()
    except Exception as exc:
        raise ValueError(
            f"Failed to instantiate analysis module '{name}': {exc}"
        )
    

def list_registered() -> dict[str, str]:
    """
    Return a dictionary of all registered analysis modules.
    
    Returns
    -------
    dict
        Mapping of module name -> class name
    """
    return {name: cls.__name__ for name, cls in ANALYSIS_REGISTRY.items()}
"""
seasouound/loader/metadata_readers.py

Registry of deployment metadata readers. Each reader class parses a specific
metadata file format to extract deployment start/end times for (optionally)
specific hydrophone serial numbers/deployment IDs.

"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from seasound.core.exceptions import SeaSoundError

logger = logging.getLogger(__name__)


@dataclass
class DeploymentWindow:
    """Standardised output from any metadata reader."""
    deploy_utc: datetime
    retrieve_utc: datetime
    location_id: Optional[str] = None
    hydrophone: Optional[str] = None


class MetadataReader(ABC):
    """
    Base class for deployment metadata readers.

    Each subclass knows how to parse a specific metadata file format
    and extract deployment/retrieval times for a given location_id and
    hydrophone.
    """
    name: str

    @abstractmethod
    def read(
        self,
        filepath: str,
        location_id: str,
        hydrophone: str,
    ) -> DeploymentWindow:
        """
        Look up deployment times for a location_id/hydrophone pair.

        Parameters
        ----------
        filepath : str
            Path to the metadata file.
        location_id : str
            Location identifier to match (can be used ot match any grouping
            variable in the deployment metadata).
        hydrophone : str
            Hydrophone serial/ID to match.

        Returns
        -------
        DeploymentWindow

        Raises
        ------
        SeaSoundError
            If the file can't be read or no matching row is found.
        """
        ...


class SeaSoundMetadataReader(MetadataReader):
    """
    SeaSound default CSV format.

    Expected columns: location_id, hydrophone, deploy_utc, retrieve_utc
    
    This is the recommended format for new projects. Create from
    the template at config/deployment_metadata_template.csv.
    """
    name = "seasound"

    def read(self, filepath, location_id, hydrophone):
        try:
            df = pd.read_csv(filepath, dtype=str)
        except Exception as exc:
            raise SeaSoundError(
                f"Could not read metadata CSV {filepath}: {exc}"
            )

        df.columns = df.columns.str.strip().str.lower()

        required = {"location_id", "hydrophone", "deploy_utc", "retrieve_utc"}
        missing = required - set(df.columns)
        if missing:
            raise SeaSoundError(
                f"Metadata file {filepath} is missing columns: "
                f"{', '.join(sorted(missing))}. "
                f"Expected SeaSound format with columns: "
                f"{', '.join(sorted(required))}"
            )

        mask = (
            (df["location_id"].str.strip().str.upper() == location_id.upper())
            & (df["hydrophone"].str.strip() == str(hydrophone))
        )
        matches = df[mask]

        if matches.empty:
            raise SeaSoundError(
                f"No row in {filepath} for location_id='{location_id}', "
                f"hydrophone='{hydrophone}'"
            )

        row = matches.iloc[0]
        return DeploymentWindow(
            deploy_utc=pd.to_datetime(row["deploy_utc"]),
            retrieve_utc=pd.to_datetime(row["retrieve_utc"]),
            location_id=location_id,
            hydrophone=hydrophone,
        )
    

class GenericExcelReader(MetadataReader):
    """
    User-configured Excel metadata reader.

    Column names are specified in the config, so this reader can
    handle arbitrary spreadsheet layouts without code changes.

    Config:
        metadata_format: "excel"
        metadata_columns:
          location_id: "Location_ID"  # column name for location/station ID
          hydrophone: "Hydrophone"
          deploy: "DateTime_deploy_UTC"
          retrieve: "DateTime_retrieve_UTC"
    """
    name = "excel"

    def __init__(self, column_map: dict):
        """
        Parameters
        ----------
        column_map : dict
            Maps logical names to actual column names in the spreadsheet.
            Required keys: "location_id", "hydrophone", "deploy", "retrieve".
        """
        required = {"location_id", "hydrophone", "deploy", "retrieve"}
        missing = required - set(column_map.keys())
        if missing:
            raise SeaSoundError(
                f"metadata_columns is missing keys: "
                f"{', '.join(sorted(missing))}. "
                f"Required: {', '.join(sorted(required))}"
            )
        self.column_map = column_map

    def read(self, filepath, location_id, hydrophone):
        try:
            df = pd.read_excel(filepath, dtype=str)
        except Exception as exc:
            raise SeaSoundError(
                f"Could not read metadata Excel {filepath}: {exc}"
            )

        df.columns = df.columns.str.strip()
        cm = self.column_map

        for logical, actual in cm.items():
            if actual not in df.columns:
                raise SeaSoundError(
                    f"Column '{actual}' (mapped from '{logical}') "
                    f"not found in {filepath}. "
                    f"Available: {', '.join(df.columns.tolist())}"
                )

        mask = (
            (df[cm["location_id"]].str.strip().str.upper()
             == location_id.upper())
            & (df[cm["hydrophone"]].str.strip()
               == str(hydrophone))
        )
        matches = df[mask]

        if matches.empty:
            raise SeaSoundError(
                f"No row in {filepath} for "
                f"{cm['location_id']}='{location_id}', "
                f"{cm['hydrophone']}='{hydrophone}'"
            )

        row = matches.iloc[0]
        return DeploymentWindow(
            deploy_utc=pd.to_datetime(row[cm["deploy"]]),
            retrieve_utc=pd.to_datetime(row[cm["retrieve"]]),
            location_id=location_id,
            hydrophone=hydrophone,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METADATA_REGISTRY: dict[str, type[MetadataReader]] = {
    "seasound": SeaSoundMetadataReader,
    "excel": GenericExcelReader,
}


def get_metadata_reader(config) -> MetadataReader:
    """
    Instantiate the configured metadata reader.

    Parameters
    ----------
    config : DeploymentConfig

    Returns
    -------
    MetadataReader
    """
    fmt = config.metadata_format
    cls = METADATA_REGISTRY.get(fmt)

    if cls is None:
        available = ", ".join(sorted(METADATA_REGISTRY.keys()))
        raise SeaSoundError(
            f"Unknown metadata_format '{fmt}'. Available: {available}"
        )

    if fmt == "excel":
        return cls(column_map=config.metadata_columns) # pyright: ignore[reportCallIssue]
    else:
        return cls()
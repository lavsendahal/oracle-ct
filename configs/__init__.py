# janus/configs/__init__.py
"""Janus Configuration - Dynamic Disease Config Loading"""

from .disease_config import (
    # Core types
    DiseaseConfig,
    ORGAN_TO_CHANNEL,
    CHANNEL_TO_ORGAN,

    # Dynamic loading functions
    load_disease_config,
    load_config_globally,
    is_config_loaded,

    # Accessors (work after load_config_globally is called)
    get_disease_config,
    get_organs_for_disease,
    get_all_disease_configs,
    get_all_diseases,
)

# Note: DISEASE_CONFIGS and ALL_DISEASES are no longer module-level constants.
# Use get_all_disease_configs() and get_all_diseases() after calling load_config_globally().

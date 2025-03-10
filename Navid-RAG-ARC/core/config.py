# This file is kept for backward compatibility
# All settings have been consolidated into core/settings.py

from .settings import Settings, get_settings

# Re-export the settings classes and functions
__all__ = ['Settings', 'get_settings']
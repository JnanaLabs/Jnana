"""
Supabase Client
Centralised Supabase client factory using supabase-py v2.

Two clients are exposed:
  get_read_client()   — uses SUPABASE_ANON_KEY  (safe for queries/reads)
  get_write_client()  — uses SUPABASE_SERVICE_ROLE_KEY (needed for upserts/writes)

Both clients are singletons (created once, reused across calls).
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from supabase import create_client, Client
from loguru import logger

load_dotenv()


def _get_url() -> str:
    url = os.getenv("SUPABASE_URL")
    if not url:
        raise EnvironmentError("SUPABASE_URL is not set in .env")
    return url


@lru_cache(maxsize=1)
def get_read_client() -> Client:
    """Supabase client with anon key — safe for read queries from the API."""
    key = os.getenv("SUPABASE_ANON_KEY")
    if not key:
        raise EnvironmentError("SUPABASE_ANON_KEY is not set in .env")
    logger.debug("Initialising Supabase read client (anon key)")
    return create_client(_get_url(), key)


@lru_cache(maxsize=1)
def get_write_client() -> Client:
    """
    Supabase client with service role key — used by the pipeline for upserts.
    Bypasses Row Level Security. NEVER expose this key to frontend clients.
    """
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        # Graceful fallback to anon key with a loud warning
        logger.warning(
            "SUPABASE_SERVICE_ROLE_KEY not set — falling back to SUPABASE_ANON_KEY. "
            "Upserts may fail if Row Level Security is enabled on vedic_chunks."
        )
        key = os.getenv("SUPABASE_ANON_KEY")
        if not key:
            raise EnvironmentError("Neither SUPABASE_SERVICE_ROLE_KEY nor SUPABASE_ANON_KEY is set in .env")
    logger.debug("Initialising Supabase write client (service role key)")
    return create_client(_get_url(), key)

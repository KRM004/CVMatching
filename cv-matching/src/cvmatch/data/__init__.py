"""Étape « données » : catalogue d'offres (Rekrute) et données fictives de test."""
from .catalog import job_texts, load_jobs, load_raw_jobs, prepare_jobs, validate_jobs

__all__ = ["job_texts", "load_jobs", "load_raw_jobs", "prepare_jobs", "validate_jobs"]

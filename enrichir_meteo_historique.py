import logging
import os
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("meteo_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ResilientMeteoExtractor:
    """Gestionnaire de requêtes vers l'API Open-Meteo avec respect des quotas
    et reprise après incident."""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        requests_per_hour_limit: int = 4900,
        pause_on_429_seconds: int = 3600,
        request_timeout: int = 15,
    ):
        self.requests_per_hour_limit = requests_per_hour_limit
        self.pause_on_429_seconds = pause_on_429_seconds
        self.request_timeout = request_timeout
        self.session = self._build_session()
        self.call_count = 0

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=10, pool_maxsize=10
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch_historical_day(
        self, lat: float, lon: float, date_str: str
    ) -> Optional[Dict[str, Any]]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "daily": "temperature_2m_max,wind_speed_10m_max,precipitation_sum",
            "timezone": "auto",
        }

        while True:
            try:
                response = self.session.get(
                    self.BASE_URL, params=params, timeout=self.request_timeout
                )

                if response.status_code == 429:
                    logger.warning(
                        f"Quota atteint (HTTP 429). Pause de {self.pause_on_429_seconds // 60} minutes "
                        "pour réinitialisation de la fenêtre horaire..."
                    )
                    time.sleep(self.pause_on_429_seconds)
                    logger.info("Reprise des requêtes après temporisation.")
                    continue

                if response.status_code == 200:
                    self.call_count += 1
                    time.sleep(0.3) # Pause technique pour lisser le trafic
                    return response.json()

                logger.error(
                    f"Erreur HTTP {response.status_code} sur ({lat}, {lon}) à la date {date_str}: {response.text}"
                )
                return None

            except requests.exceptions.RequestException as exc:
                logger.error(f"Erreur réseau sur ({lat}, {lon}) : {exc}")
                return None

    def process_dataframe(
        self,
        df: pd.DataFrame,
        output_csv_path: str,
        checkpoint_freq: int = 100,
    ) -> None:
        start_index = 0

        # Reprise sur incident (Checkpointing)
        if os.path.exists(output_csv_path):
            existing_df = pd.read_csv(output_csv_path)
            start_index = len(existing_df)
            logger.info(
                f"Fichier existant détecté : reprise à l'index {start_index}/{len(df)}."
            )
            records: List[Dict[str, Any]] = existing_df.to_dict(orient="records")
        else:
            records = []

        for idx in range(start_index, len(df)):
            row = df.iloc[idx]
            lat = row["latitude"]
            lon = row["longitude"]
            date_val = str(row["acq_date"]).split(" ")[0]  # Format ISO : YYYY-MM-DD

            data = self.fetch_historical_day(lat, lon, date_val)

            if data and "daily" in data:
                daily = data["daily"]
                tmax = daily.get("temperature_2m_max", [None])[0] if daily.get("temperature_2m_max") else None
                wind = daily.get("wind_speed_10m_max", [None])[0] if daily.get("wind_speed_10m_max") else None
                precip = daily.get("precipitation_sum", [None])[0] if daily.get("precipitation_sum") else None
                # Estimation de l'humidité moyenne (souvent absente des archives gratuites)
                h_mean = max(20.0, 100 - (tmax * 1.5)) if tmax else None
            else:
                tmax, wind, precip, h_mean = None, None, None, None

            record = {
                **row.to_dict(),
                "t_max": tmax,
                "wind_max": wind,
                "precip_sum": precip,
                "h_mean": h_mean
            }
            records.append(record)

            if (idx + 1) % checkpoint_freq == 0 or (idx + 1) == len(df):
                pd.DataFrame(records).to_csv(output_csv_path, index=False)
                logger.info(
                    f"Progression : {idx + 1}/{len(df)} lignes traitées et sauvegardées."
                )

        logger.info(
            f"Extraction terminée avec succès. Fichier sauvegardé : {output_csv_path}"
        )


if __name__ == "__main__":
    # 1. Spécifie le fichier d'échantillon créé précédemment (8000 lignes)
    fichier_entree = "dataset_echantillon_ml.csv"
    fichier_sortie = "dataset_historique_meteo.csv"
    
    if os.path.exists(fichier_entree):
        df_echantillon = pd.read_csv(fichier_entree)
        extracteur = ResilientMeteoExtractor(pause_on_429_seconds=3600)
        extracteur.process_dataframe(df_echantillon, fichier_sortie, checkpoint_freq=100)
    else:
        logger.error(f"Fichier {fichier_entree} introuvable. Veuillez exécuter echantillonnage.py d'abord.")
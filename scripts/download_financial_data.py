from enso_commodities.config import project_root
from enso_commodities.download import download_all

if __name__ == "__main__":
    print(
        download_all(
            raw_root=project_root() / "data" / "financial" / "raw",
            config_path=project_root() / "config" / "financial_sources.yaml",
        )
    )

from enso_commodities.config import project_root
from enso_commodities.download import download_all

if __name__ == "__main__":
    print(
        download_all(
            raw_root=project_root() / "data" / "mechanisms" / "palm_oil" / "raw",
            config_path=project_root() / "config" / "palm_oil_sources.yaml",
        )
    )

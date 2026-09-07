import sys
from pathlib import Path

from enso_commodities.config import project_root
from enso_commodities.publication import build_publication_bundle

if __name__ == "__main__":
    check = "--check" in sys.argv[1:]
    values = [value for value in sys.argv[1:] if value != "--check"]
    bundle = Path(values[0]) if values else None
    if check and bundle is None:
        candidates = sorted((project_root() / "reports" / "artifacts").glob("????-??-??"))
        if not candidates:
            raise SystemExit("No published result bundles found")
        bundle = candidates[-1]
    print(build_publication_bundle(check=check, bundle=bundle))

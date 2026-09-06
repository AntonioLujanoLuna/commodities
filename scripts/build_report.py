import sys

from enso_commodities.reporting import build_current_results

if __name__ == "__main__":
    print(build_current_results(check="--check" in sys.argv[1:]))

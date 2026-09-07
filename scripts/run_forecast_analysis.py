from enso_commodities.forecast_analysis import run_forecast_analysis

if __name__ == "__main__":
    output = run_forecast_analysis()
    print(f"Wrote pseudo-out-of-sample forecast outputs to {output}")

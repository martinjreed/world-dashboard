import io
import sys
import textwrap
import pandas as pd
from pandas_datareader import wb

START_YEAR, END_YEAR = 1990, 2023

# World Bank indicators (keep names stable for your dashboard)
WB_INDICATORS = {
    "EN.POP.DNST": "population_density",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    # CO2 per capita sometimes fails via WB API; we add an OWID fallback below
    "IT.NET.USER.ZS": "internet_users_pct",
    "EG.FEC.RNEW.ZS": "renewables_pct_final_energy",
    "SP.DYN.LE00.IN": "life_expectancy_years",
    "SP.URB.TOTL.IN.ZS": "urban_pop_pct",
}

# OWID fallback for CO2 per-capita
OWID_CO2_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
OWID_CO2_COL = "co2_per_capita"  # tons per person
OWID_CO2_NAME = "co2_per_capita_tons"

def get_countries_table():
    """Get WB countries, filter out aggregates, keep ISO-3 and canonical names."""
    c = wb.get_countries()
    c = c[c["region"] != "Aggregates"][["name", "iso3c"]]
    c = c.rename(columns={"name": "country", "iso3c": "iso3"})
    return c

def fetch_wb_indicator(code, friendly, countries_df):
    """
    Robust WB fetch: request by 'all' countries, then left-join to countries_df
    to attach ISO-3 and discard aggregates. Returns long df: iso3, country, year, indicator, value.
    """
    print(f"Fetching WB {code} -> {friendly}")
    try:
        df = wb.download(
            indicator=code,
            country="all",
            start=START_YEAR,
            end=END_YEAR,
        ).reset_index()
    except Exception as e:
        print(f"  ⚠️ WB failed for {code}: {e}")
        return pd.DataFrame([])

    # WB returns columns: country, year, <code>
    if not {"country", "year", code}.issubset(df.columns):
        print(f"  ⚠️ WB returned unexpected columns for {code}: {df.columns.tolist()}")
        return pd.DataFrame([])

    # Attach ISO-3 and drop aggregates (only rows that match countries_df)
    df = df.merge(countries_df, on="country", how="left")
    df = df.dropna(subset=["iso3"])

    # Tidy to long format
    df = df.rename(columns={code: "value"})
    df["indicator"] = friendly
    df = df[["iso3", "country", "year", "indicator", "value"]]
    df = df.dropna(subset=["value"])
    return df

def fetch_owid_co2(countries_df):
    """
    Fallback for CO2 per capita using OWID. Returns long df aligned with the same schema.
    """
    print(f"Fetching OWID CO2 fallback -> {OWID_CO2_NAME}")
    try:
        ow = pd.read_csv(OWID_CO2_URL)
    except Exception as e:
        print(f"  ⚠️ OWID CO2 fetch failed: {e}")
        return pd.DataFrame([])

    # OWID columns: country, iso_code, year, co2_per_capita ...
    cols_needed = {"country", "iso_code", "year", OWID_CO2_COL}
    if not cols_needed.issubset(ow.columns):
        print(f"  ⚠️ OWID CO2 missing expected columns: {ow.columns.tolist()}")
        return pd.DataFrame([])

    # Filter to countries in our list (drop OWID aggregates like 'World', 'Asia', etc.)
    ow = ow.rename(columns={"iso_code": "iso3"})
    ow = ow.merge(countries_df[["iso3"]], on="iso3", how="inner")

    ow = ow.rename(columns={OWID_CO2_COL: "value"})
    ow["indicator"] = OWID_CO2_NAME
    ow = ow[["iso3", "country", "year", "indicator", "value"]]
    ow = ow.dropna(subset=["value"])
    ow = ow[(ow["year"] >= START_YEAR) & (ow["year"] <= END_YEAR)]
    return ow

def main():
    countries_df = get_countries_table()

    frames = []

    # 1) World Bank indicators (robust join by country name -> iso3)
    for code, friendly in WB_INDICATORS.items():
        df = fetch_wb_indicator(code, friendly, countries_df)
        if df.empty:
            print(f"  ⚠️ No data for {code} ({friendly})")
        else:
            frames.append(df)

    # 2) CO2 per-capita: try WB first; if empty, use OWID fallback
    co2_wb = fetch_wb_indicator("EN.ATM.CO2E.PC", "co2_per_capita_tons", countries_df)
    if co2_wb.empty:
        co2_owid = fetch_owid_co2(countries_df)
        if co2_owid.empty:
            print("  ⚠️ Could not fetch CO2 per capita from WB or OWID; skipping.")
        else:
            frames.append(co2_owid)
    else:
        frames.append(co2_wb)

    if not frames:
        raise RuntimeError("No datasets retrieved. Check network or try again later.")

    long_df = pd.concat(frames, ignore_index=True)
    # Clean and sort
    long_df = long_df.dropna(subset=["value"])
    long_df = long_df.sort_values(["indicator", "country", "year"]).reset_index(drop=True)

    # Build latest-year wide table
    latest = (
        long_df.dropna(subset=["value"])
               .sort_values(["iso3", "indicator", "year"])
               .groupby(["iso3", "country", "indicator"], as_index=False)
               .last()
    )
    wide = latest.pivot_table(index=["iso3", "country"], columns="indicator", values="value").reset_index()

    long_df.to_csv("world_data_long.csv", index=False)
    wide.to_csv("world_data_wide_latest.csv", index=False)

    print("\n✅ Wrote:")
    print(" - world_data_long.csv   (iso3, country, year, indicator, value)")
    print(" - world_data_wide_latest.csv  (one row per country, latest values)")
    print("\nIndicators included:", sorted(long_df['indicator'].unique()))

if __name__ == "__main__":
    main()

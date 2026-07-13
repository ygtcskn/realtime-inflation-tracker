import re
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_OUTPUT, OUTPUT_GRAPHS

warnings.filterwarnings("ignore")

BASE_DIR = DATA_OUTPUT / "B_LSTM"
PERM_INPUT_CSV = DATA_OUTPUT / "B_LSTM_perm" / "B_LSTM_permutation_importance_rolling_raw.csv"

GRAPH7_PNG = OUTPUT_GRAPHS / "Graph_7_Permutation_Period_Comparison.png"
GRAPH7_PDF = OUTPUT_GRAPHS / "Graph_7_Permutation_Period_Comparison.pdf"

TOP_N = 20

PERIODS = {
    "Pandemic Shock": ("2020-01-01", "2020-12-31"),
    "Inflation Surge": ("2021-01-01", "2023-12-31"),
    "Normalisation": ("2024-01-01", "2025-12-31"),
}

PERIODS_ORDER = ["Pandemic Shock", "Inflation Surge", "Normalisation"]

NEUTRAL_FIG = "#FFFFFF"
NEUTRAL_TEXT = "#222222"

RENAME_DICT = {
    # ----- Financial variables (no GT suffix) -----
    "v_oil": "Oil",
    "v_us10y": "US 10Y Yield",
    "v_fxrate": "FX Rate",
    "v_vix": "VIX",
    "v_gold": "Gold",
    "v_stock": "Stock Index",

    # ----- Google Trends categories -----
    "events_listings": "Events & Listings (GT)",
    "travel": "Travel (GT)",
    "sports": "Sports (GT)",
    "hotels_accommodations": "Hotels & Accommodations (GT)",
    "travel_agencies_services": "Travel Agencies & Services (GT)",
    "tourist_destinations": "Tourist Destinations (GT)",
    "cruises_charters": "Cruises & Charters (GT)",
    "visual_art_design": "Visual Art & Design (GT)",
    "business_industrial": "Business & Industrial (GT)",
    "corporate_events": "Corporate Events (GT)",
    "computers_electronics": "Computers & Electronics (GT)",
    "manufacturing": "Manufacturing (GT)",
    "politics": "Politics (GT)",
    "business_news": "Business News (GT)",
    "business_operations": "Business & Operations (GT)",
    "finance": "Finance (GT)",
    "real_estate": "Real Estate (GT)",
    "education": "Education (GT)",
    "newspapers": "Newspapers (GT)",
    "computer_hardware": "Computer Hardware (GT)",
    "government": "Government (GT)",
    "entertainment_industry": "Entertainment Industry (GT)",
    "gifts_special_event_items": "Gifts & Special Event Items (GT)",
    "news": "News (GT)",
    "pets_animals": "Pets & Animals (GT)",
    "banking": "Banking (GT)",
    "agriculture_forestry": "Agriculture & Forestry (GT)",
    "apartments_residential_rentals": "Apartments & Residential Rentals (GT)",
    "car_rental_taxi_services": "Car Rental & Taxi Services (GT)",
    "beauty_fitness": "Beauty & Fitness (GT)",
    "retail_trade": "Retail Trade (GT)",
    "unwanted_body_facial_hair_removal": "Body & Facial Hair Removal (GT)",
    "entertainment_media": "Entertainment & Media (GT)",
    "rail_transport": "Rail Transport (GT)",
    "autos_vehicles": "Autos & Vehicles (GT)",
    "coupons_discount_offers": "Coupons & Discount Offers (GT)",
    "pharmaceuticals_biotech": "Pharmaceuticals & Biotech (GT)",
    "resumes_portfolios": "Resumes & Portfolios (GT)",
    "computer_security": "Computer Security (GT)",
    "social_issues_advocacy": "Social Issues & Advocacy (GT)",
    "real_estate_listings": "Real Estate Listings (GT)",
    "food_production": "Food Production (GT)",
    "marketing_services": "Marketing Services (GT)",
    "industrial_materials_equipment": "Industrial Materials & Equipment (GT)",
    "legal_services": "Legal Services (GT)",
    "footwear": "Footwear (GT)",
    "web_hosting_domain_registration": "Web Hosting & Domain Registration (GT)",
    "home_financing": "Home Financing (GT)",
    "office_services": "Office Services (GT)",
    "world_news": "World News (GT)",
    "vehicle_brands": "Vehicle Brands (GT)",
    "performing_arts": "Performing Arts (GT)",
    "consumer_resources": "Consumer Resources (GT)",
    "metals_mining": "Metals & Mining (GT)",
    "distribution_logistics": "Distribution & Logistics (GT)",
    "antiques_collectibles": "Antiques & Collectibles (GT)",
    "vehicle_fuels_lubricants": "Vehicle Fuels & Lubricants (GT)",
    "social_services": "Social Services (GT)",
    "renewable_alt_energy": "Renewable & Alt Energy (GT)",
    "programming": "Programming (GT)",
    "jobs": "Jobs (GT)",

    # ----- Google Trends topics (t_ prefix) -----
    "t_natural_gas": "Natural Gas (GT)",
    "t_interest_rate": "Interest Rate (GT)",
    "t_export": "Export (GT)",
    "t_savings": "Savings (GT)",
    "t_jobs": "Jobs (GT)",
    "t_purchasing_power": "Purchasing Power (GT)",
    "t_exchange_rate": "Exchange Rate (GT)",
    "t_bankruptcy": "Bankruptcy (GT)",
    "t_house_price_index": "House Price Index (GT)",
    "t_central_bank": "Central Bank (GT)",
    "t_student_loan": "Student Loan (GT)",
    "t_restaurant": "Restaurant (GT)",
    "t_baggage": "Baggage (GT)",
    "t_salary": "Salary (GT)",
    "t_petroleum": "Petroleum (GT)",
    "t_strike_action": "Strike Action (GT)",
    "t_auto_insurance": "Auto Insurance (GT)",
    "t_real_estate": "Real Estate (GT)",
    "t_interest": "Interest (GT)",
    "t_liquidation": "Liquidation (GT)",
    "t_housing": "Housing (GT)",
    "t_price": "Price (GT)",
    "t_currency": "Currency (GT)",
    "t_consumer_price_index": "Consumer Price Index (GT)",
    "t_public_transport": "Public Transport (GT)",
}

CATEGORY_COLORS = {
    "AR Lags": "#e74c3c",
    "Country Dummies": "#95a5a6",
    "Financial Variables": "#F29900",
    "Google Trends": "#4285F4",
}

plt.rcParams.update({
    "figure.facecolor": NEUTRAL_FIG,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D0D0D0",
    "axes.labelcolor": NEUTRAL_TEXT,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": NEUTRAL_TEXT,
    "grid.color": "#E6E6E6",
    "grid.alpha": 0.75,
    "grid.linewidth": 0.9,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#D0D0D0",
    "legend.facecolor": "#FFFFFF",
})


def save_figure(fig, png_path, pdf_path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)


def get_base_variable(name):
    match = re.match(r"^(.+)_w([1-4])$", str(name))
    return match.group(1) if match else name


def is_exogenous(name):
    return not str(name).startswith("C_") and not str(name).startswith("infl_yoy_lag")


def pretty_name(name):
    return RENAME_DICT.get(name, name)


def categorize(name):
    name = str(name)

    if name.startswith("C_"):
        return "Country Dummies"

    if name.startswith("infl_yoy_lag"):
        return "AR Lags"

    if name.startswith("v_"):
        return "Financial Variables"

    return "Google Trends"


def assign_period(date):
    date = pd.Timestamp(date)

    for name, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return name

    return "Other"


def plot_graph_7_permutation_period_comparison():
    if not PERM_INPUT_CSV.exists():
        return

    period_df = pd.read_csv(PERM_INPUT_CSV)

    if "date" in period_df.columns:
        period_df["date"] = pd.to_datetime(period_df["date"])

        if "period" not in period_df.columns:
            period_df["period"] = period_df["date"].apply(assign_period)

    if "base_variable" not in period_df.columns:
        if "feature" not in period_df.columns:
            return

        period_df["base_variable"] = period_df["feature"].apply(get_base_variable)

    if "category" not in period_df.columns:
        period_df["category"] = period_df["base_variable"].apply(categorize)

    if "mean_rmse_increase" in period_df.columns:
        value_col = "mean_rmse_increase"
    elif "total_rmse_increase" in period_df.columns:
        value_col = "total_rmse_increase"
    else:
        numeric_cols = [col for col in period_df.columns if pd.api.types.is_numeric_dtype(period_df[col])]

        if not numeric_cols:
            return

        value_col = numeric_cols[0]

    period_base = (
        period_df.groupby(["period", "base_variable", "category"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "total_value"})
    )

    period_base = period_base[period_base["base_variable"].apply(is_exogenous)].copy()
    periods_list = [period for period in PERIODS_ORDER if period in period_base["period"].values]

    if not periods_list:
        return

    top_data_by_period = {}

    for period in periods_list:
        data = (
            period_base[period_base["period"] == period]
            .sort_values("total_value", ascending=False)
            .head(TOP_N)
            .copy()
        )

        top_data_by_period[period] = data

    fig, axes = plt.subplots(
        1,
        len(periods_list),
        figsize=(7 * len(periods_list), 8),
        sharey=False,
        facecolor=NEUTRAL_FIG
    )

    if len(periods_list) == 1:
        axes = [axes]

    for i, period in enumerate(periods_list):
        data = top_data_by_period[period].copy()

        data["label"] = data["base_variable"].apply(pretty_name)

        colors = [CATEGORY_COLORS.get(category, "#7f8c8d") for category in data["category"]]

        axes[i].barh(
            data["label"].values[::-1],
            data["total_value"].values[::-1],
            color=colors[::-1],
            edgecolor="white"
        )

        axes[i].set_title(period, fontsize=13)
        axes[i].grid(axis="x", alpha=0.3)
        axes[i].set_xlabel("Mean RMSE Increase", fontsize=11)
        axes[i].tick_params(axis="y", labelsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_figure(fig, GRAPH7_PNG, GRAPH7_PDF)


if __name__ == "__main__":
    OUTPUT_GRAPHS.mkdir(parents=True, exist_ok=True)
    plot_graph_7_permutation_period_comparison()
    print("Done. Graph saved.")
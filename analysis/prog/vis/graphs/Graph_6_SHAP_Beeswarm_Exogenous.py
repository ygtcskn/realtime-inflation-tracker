import re
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import DATA_FINAL, DATA_OUTPUT, OUTPUT_GRAPHS

warnings.filterwarnings("ignore")

BASE_DIR = DATA_OUTPUT / "B_LSTM"
SHAP_CSV_DIR = BASE_DIR / "shap_rolling"
SHAP_RAW_PATH = SHAP_CSV_DIR / "B_LSTM_shap_rolling_raw.csv"
SHAP_DATA_PATH = DATA_FINAL / "B_panel.csv"

GRAPH6_PNG = OUTPUT_GRAPHS / "Graph_6_SHAP_Beeswarm_Exogenous.png"
GRAPH6_PDF = OUTPUT_GRAPHS / "Graph_6_SHAP_Beeswarm_Exogenous.pdf"

TOP_N = 20

BG_COLOR = "#FFFFFF"
NEUTRAL_FIG = "#FFFFFF"
NEUTRAL_EDGE = "#D0D0D0"
NEUTRAL_TEXT = "#222222"

RENAME_DICT = {
    "v_oil": r"Oil$^{*}$",
    "v_us10y": r"US 10Y Yield$^{*}$",
    "v_fxrate": r"FX Rate$^{*}$",
    "v_vix": r"VIX$^{*}$",
    "v_gold": r"Gold$^{*}$",
    "v_stock": r"Stock Index$^{*}$",

    "events_listings": "Events & Listings",
    "travel": "Travel",
    "sports": "Sports",
    "hotels_accommodations": "Hotels & Accommodations",
    "travel_agencies_services": "Travel Agencies & Services",
    "tourist_destinations": "Tourist Destinations",
    "cruises_charters": "Cruises & Charters",
    "visual_art_design": "Visual Art & Design",
    "business_industrial": "Business & Industrial",
    "corporate_events": "Corporate Events",
    "computers_electronics": "Computers & Electronics",
    "manufacturing": "Manufacturing",
    "politics": "Politics",
    "business_news": "Business News",
    "business_operations": "Business & Operations",
    "finance": "Finance",
    "real_estate": "Real Estate",
    "t_natural_gas": "Natural Gas",
    "t_interest_rate": "Interest Rate",
    "t_export": "Export",
    "education": "Education",
    "newspapers": "Newspapers",
    "computer_hardware": "Computer Hardware",
    "government": "Government",
    "entertainment_industry": "Entertainment Industry",
    "t_savings": "Savings",
    "t_jobs": "Jobs",
    "t_purchasing_power": "Purchasing Power",
    "gifts_special_event_items": "Gifts & Special Event Items",
    "news": "News",
    "pets_animals": "Pets & Animals",
    "banking": "Banking",
    "agriculture_forestry": "Agriculture & Forestry",
    "t_exchange_rate": "Exchange Rate",
    "t_bankruptcy": "Bankruptcy",
    "apartments_residential_rentals": "Apartments & Residential Rentals",
    "t_house_price_index": "House Price Index",
    "car_rental_taxi_services": "Car Rental & Taxi Services",
    "beauty_fitness": "Beauty & Fitness",
    "retail_trade": "Retail Trade",
    "t_central_bank": "Central Bank",
    "unwanted_body_facial_hair_removal": "Body & Facial Hair Removal",
    "entertainment_media": "Entertainment & Media",
    "rail_transport": "Rail Transport",
    "autos_vehicles": "Autos & Vehicles",
    "coupons_discount_offers": "Coupons & Discount Offers",
    "pharmaceuticals_biotech": "Pharmaceuticals & Biotech",
    "t_student_loan": "Student Loan",
    "resumes_portfolios": "Resumes & Portfolios",
    "computer_security": "Computer Security",
    "social_issues_advocacy": "Social Issues & Advocacy",
    "real_estate_listings": "Real Estate Listings",
    "food_production": "Food Production",
    "t_electricity": "Electricity",
    "fuel_economy_gas_prices": "Fuel Economy & Gas Prices",
    "marketing_services": "Marketing Services",
    "industrial_materials_equipment": "Industrial Materials & Equipment",
    "t_restaurant": "Restaurant",
    "legal_services": "Legal Services",
    "t_baggage": "Baggage",
    "footwear": "Footwear",
    "t_salary": "Salary",
    "web_hosting_domain_registration": "Web Hosting & Domain Registration",
    "t_petroleum": "Petroleum",
    "home_financing": "Home Financing",
    "t_strike_action": "Strike Action",
    "t_auto_insurance": "Auto Insurance",
    "office_services": "Office Services",
}

plt.rcParams.update({
    "figure.facecolor": NEUTRAL_FIG,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": NEUTRAL_EDGE,
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


def top_explanation(shap_values, feature_values, names, n=TOP_N):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:n]

    return shap.Explanation(
        values=shap_values[:, top_idx],
        data=feature_values[:, top_idx],
        feature_names=[pretty_name(names[i]) for i in top_idx],
    )


def plot_graph_6_shap_beeswarm_exogenous():
    if not SHAP_RAW_PATH.exists() or not SHAP_DATA_PATH.exists():
        return

    raw = pd.read_csv(SHAP_RAW_PATH)
    raw["date"] = pd.to_datetime(raw["date"])

    meta_cols = ["date", "country", "period"]
    feature_cols = [col for col in raw.columns if col not in meta_cols]
    shap_values = raw[feature_cols].values

    panel = pd.read_csv(SHAP_DATA_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    panel_keep = ["date", "country"] + [col for col in feature_cols if col in panel.columns]
    panel = panel[panel_keep].drop_duplicates(subset=["date", "country"])

    merged = raw[["date", "country"]].merge(panel, on=["date", "country"], how="left")
    feature_values = merged[feature_cols].values.astype(np.float32)
    feature_values = np.nan_to_num(feature_values, nan=0.0)

    base_var_map = {}

    for i, feature in enumerate(feature_cols):
        base_var_map.setdefault(get_base_variable(feature), []).append(i)

    base_names = list(base_var_map.keys())

    shap_base = np.zeros((shap_values.shape[0], len(base_names)))
    feature_base = np.zeros((feature_values.shape[0], len(base_names)))

    for i, base_name in enumerate(base_names):
        idx = base_var_map[base_name]
        shap_base[:, i] = shap_values[:, idx].sum(axis=1)
        feature_base[:, i] = feature_values[:, idx].mean(axis=1)

    exog_idx = [i for i, base_name in enumerate(base_names) if is_exogenous(base_name)]

    if not exog_idx:
        return

    exog_names = [base_names[i] for i in exog_idx]
    shap_exog = shap_base[:, exog_idx]
    feature_exog = feature_base[:, exog_idx]

    explanation = top_explanation(shap_exog, feature_exog, exog_names)

    fig = plt.figure(figsize=(8.27, 6.2), facecolor=NEUTRAL_FIG)
    ax = plt.gca()

    ax.set_facecolor("#f0f0f0")
    ax.grid(True, axis="x", color="white", linewidth=1.0)
    ax.grid(True, axis="y", color="white", linewidth=1.0)
    ax.set_axisbelow(True)

    shap.plots.beeswarm(explanation, max_display=TOP_N, show=False)

    ax.set_title("")
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, axis="x", color="white", linewidth=1.0)
    ax.grid(True, axis="y", color="white", linewidth=1.0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, GRAPH6_PNG, GRAPH6_PDF)


if __name__ == "__main__":
    OUTPUT_GRAPHS.mkdir(parents=True, exist_ok=True)
    plot_graph_6_shap_beeswarm_exogenous()
    print("Done. Graph saved.")

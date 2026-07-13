import re
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_OUTPUT, OUTPUT_GRAPHS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PERM_INPUT_CSV = DATA_OUTPUT / "B_LSTM_perm" / "B_LSTM_permutation_importance_rolling_raw.csv"

GRAPH72_PNG = OUTPUT_GRAPHS / "Graph_7_2_Permutation_Period_Comparison.png"
GRAPH72_PDF = OUTPUT_GRAPHS / "Graph_7_2_Permutation_Period_Comparison.pdf"

CLIP_NEGATIVE_IMPORTANCE = False

PERIODS = {
    "Pandemic Shock": ("2020-01-01", "2020-12-31"),
    "Inflation Surge": ("2021-01-01", "2023-12-31"),
    "Normalisation": ("2024-01-01", "2025-12-31"),
}

PERIODS_ORDER = ["Pandemic Shock", "Inflation Surge", "Normalisation"]

# ---------------------------------------------------------------------
# COLORS / BACKGROUND
# ---------------------------------------------------------------------
NEUTRAL_FIG = "#FFFFFF"   # outside figure background
BG_COLOR = "#F0F0F0"      # inside graph background
NEUTRAL_TEXT = "#222222"

# ---------------------------------------------------------------------
# LAYOUT TUNING
# ---------------------------------------------------------------------
POINT_MARGIN = 0.25       # equal space: left border to first point = right border to last point
LABEL_X_OFFSET = 0.34     # labels are placed outside the right border
SAVE_PAD = 0.002
RIGHT_MARGIN = 0.76       # lower = more white space for labels on the right

# ---------------------------------------------------------------------
# SECTOR MAPPING
# ---------------------------------------------------------------------
SECTOR_MAPPING = {
    # ---------- Macro & Finance ----------
    "v_oil": "Macro & Finance",
    "v_gold": "Macro & Finance",
    "v_copper": "Macro & Finance",
    "v_us10y": "Macro & Finance",
    "v_stock": "Macro & Finance",
    "v_vix": "Macro & Finance",
    "v_fxrate": "Macro & Finance",
    "finance": "Macro & Finance",
    "banking": "Macro & Finance",
    "t_interest_rate": "Macro & Finance",
    "t_central_bank": "Macro & Finance",
    "t_exchange_rate": "Macro & Finance",
    "t_bankruptcy": "Macro & Finance",
    "home_financing": "Macro & Finance",
    "t_auto_insurance": "Macro & Finance",
    "auto_financing": "Macro & Finance",
    "bankruptcy": "Macro & Finance",
    "commercial_lending": "Macro & Finance",
    "credit_lending": "Macro & Finance",
    "economy_news": "Macro & Finance",
    "insurance": "Macro & Finance",
    "t_consumer_price_index": "Macro & Finance",
    "t_cost_of_living": "Macro & Finance",
    "t_crisis": "Macro & Finance",
    "t_currency": "Macro & Finance",
    "t_debt": "Macro & Finance",
    "t_economic_crisis": "Macro & Finance",
    "t_financial_crisis": "Macro & Finance",
    "t_foreign_exchange_market": "Macro & Finance",
    "t_inflation": "Macro & Finance",
    "t_interest": "Macro & Finance",
    "t_investment": "Macro & Finance",
    "t_liquidation": "Macro & Finance",
    "t_monetary_policy": "Macro & Finance",
    "t_price": "Macro & Finance",
    "t_recession": "Macro & Finance",

    # ---------- Travel & Tourism ----------
    "travel": "Travel & Tourism",
    "hotels_accommodations": "Travel & Tourism",
    "travel_agencies_services": "Travel & Tourism",
    "tourist_destinations": "Travel & Tourism",
    "cruises_charters": "Travel & Tourism",
    "car_rental_taxi_services": "Travel & Tourism",
    "rail_transport": "Travel & Tourism",
    "autos_vehicles": "Travel & Tourism",
    "t_baggage": "Travel & Tourism",
    "aviation": "Travel & Tourism",
    "boats_watercraft": "Travel & Tourism",
    "carpooling_ridesharing": "Travel & Tourism",
    "luggage_travel_accessories": "Travel & Tourism",
    "maritime_transport": "Travel & Tourism",
    "parking": "Travel & Tourism",
    "t_public_transport": "Travel & Tourism",
    "timeshares_vacation_properties": "Travel & Tourism",
    "trucks_suvs": "Travel & Tourism",
    "urban_transport": "Travel & Tourism",
    "vehicle_brands": "Travel & Tourism",
    "vehicle_fuels_lubricants": "Travel & Tourism",
    "vehicle_licensing_registration": "Travel & Tourism",
    "vehicle_parts_accessories": "Travel & Tourism",
    "vehicle_shopping": "Travel & Tourism",
    "car_electronics": "Travel & Tourism",

    # ---------- Real Estate ----------
    "real_estate": "Real Estate",
    "real_estate_listings": "Real Estate",
    "apartments_residential_rentals": "Real Estate",
    "t_house_price_index": "Real Estate",
    "architecture": "Real Estate",
    "building_materials_supplies": "Real Estate",
    "construction_consulting_contracting": "Real Estate",
    "construction_maintenance": "Real Estate",
    "flooring": "Real Estate",
    "home_improvement": "Real Estate",
    "home_insurance": "Real Estate",
    "moving_relocation": "Real Estate",
    "property_management": "Real Estate",
    "real_estate_agencies": "Real Estate",
    "t_affordable_housing": "Real Estate",
    "t_housing": "Real Estate",
    "t_real_estate": "Real Estate",
    "t_rent": "Real Estate",

    # ---------- Labour & Income ----------
    "t_jobs": "Labour & Income",
    "t_salary": "Labour & Income",
    "t_purchasing_power": "Labour & Income",
    "t_savings": "Labour & Income",
    "t_student_loan": "Labour & Income",
    "t_strike_action": "Labour & Income",
    "resumes_portfolios": "Labour & Income",
    "college_financing": "Labour & Income",
    "developer_jobs": "Labour & Income",
    "job_listings": "Labour & Income",
    "jobs": "Labour & Income",
    "t_minimum_wage": "Labour & Income",
    "t_recruitment": "Labour & Income",
    "t_unemployment": "Labour & Income",
    "t_unemployment_benefits": "Labour & Income",
    "t_wage": "Labour & Income",

    # ---------- Business & Industry ----------
    "business_industrial": "Business & Industry",
    "business_news": "Business & Industry",
    "business_operations": "Business & Industry",
    "corporate_events": "Business & Industry",
    "manufacturing": "Business & Industry",
    "industrial_materials_equipment": "Business & Industry",
    "retail_trade": "Business & Industry",
    "food_production": "Business & Industry",
    "agriculture_forestry": "Business & Industry",
    "t_export": "Business & Industry",
    "marketing_services": "Business & Industry",
    "office_services": "Business & Industry",
    "legal_services": "Business & Industry",
    "accounting_auditing": "Business & Industry",
    "advertising_marketing": "Business & Industry",
    "agricultural_equipment": "Business & Industry",
    "agrochemicals": "Business & Industry",
    "animal_products_services": "Business & Industry",
    "aquaculture": "Business & Industry",
    "business_services": "Business & Industry",
    "chemicals_industry": "Business & Industry",
    "civil_engineering": "Business & Industry",
    "coatings_adhesives": "Business & Industry",
    "commercial_vehicles": "Business & Industry",
    "consulting": "Business & Industry",
    "distribution_logistics": "Business & Industry",
    "dyes_pigments": "Business & Industry",
    "forestry": "Business & Industry",
    "freight_trucking": "Business & Industry",
    "import_export": "Business & Industry",
    "legal": "Business & Industry",
    "metals_mining": "Business & Industry",
    "office_supplies": "Business & Industry",
    "outsourcing": "Business & Industry",
    "packaging": "Business & Industry",
    "professional_trade_associations": "Business & Industry",
    "t_invoice": "Business & Industry",
    "t_lawyer": "Business & Industry",
    "textiles_nonwovens": "Business & Industry",
    "transportation_logistics": "Business & Industry",

    # ---------- Tech & Media ----------
    "computers_electronics": "Tech & Media",
    "computer_hardware": "Tech & Media",
    "computer_security": "Tech & Media",
    "web_hosting_domain_registration": "Tech & Media",
    "entertainment_industry": "Tech & Media",
    "entertainment_media": "Tech & Media",
    "news": "Tech & Media",
    "newspapers": "Tech & Media",
    "visual_art_design": "Tech & Media",
    "acting_theater": "Tech & Media",
    "arts_entertainment": "Tech & Media",
    "book_retailers": "Tech & Media",
    "books_literature": "Tech & Media",
    "cad_cam": "Tech & Media",
    "computer_servers": "Tech & Media",
    "computer_video_games": "Tech & Media",
    "consumer_electronics": "Tech & Media",
    "customer_relationship_management_crm": "Tech & Media",
    "data_management": "Tech & Media",
    "development_tools": "Tech & Media",
    "enterprise_resource_planning_erp": "Tech & Media",
    "enterprise_technology": "Tech & Media",
    "gps_navigation": "Tech & Media",
    "internet_telecom": "Tech & Media",
    "movies": "Tech & Media",
    "music_audio": "Tech & Media",
    "performing_arts": "Tech & Media",
    "printing_publishing": "Tech & Media",
    "programming": "Tech & Media",
    "search_engine_optimization_marketing": "Tech & Media",
    "software": "Tech & Media",
    "tv_video": "Tech & Media",
    "world_news": "Tech & Media",

    # ---------- Consumer & Lifestyle ----------
    "events_listings": "Consumer & Lifestyle",
    "sports": "Consumer & Lifestyle",
    "gifts_special_event_items": "Consumer & Lifestyle",
    "beauty_fitness": "Consumer & Lifestyle",
    "footwear": "Consumer & Lifestyle",
    "coupons_discount_offers": "Consumer & Lifestyle",
    "t_restaurant": "Consumer & Lifestyle",
    "pets_animals": "Consumer & Lifestyle",
    "unwanted_body_facial_hair_removal": "Consumer & Lifestyle",
    "alcoholic_beverages": "Consumer & Lifestyle",
    "antiques_collectibles": "Consumer & Lifestyle",
    "apparel": "Consumer & Lifestyle",
    "card_games": "Consumer & Lifestyle",
    "consumer_advocacy_protection": "Consumer & Lifestyle",
    "consumer_resources": "Consumer & Lifestyle",
    "dating_personals": "Consumer & Lifestyle",
    "fast_food": "Consumer & Lifestyle",
    "fitness": "Consumer & Lifestyle",
    "food_drink": "Consumer & Lifestyle",
    "games": "Consumer & Lifestyle",
    "gifts": "Consumer & Lifestyle",
    "grocery_food_retailers": "Consumer & Lifestyle",
    "hobbies_leisure": "Consumer & Lifestyle",
    "home_appliances": "Consumer & Lifestyle",
    "home_furnishings": "Consumer & Lifestyle",
    "home_garden": "Consumer & Lifestyle",
    "luxury_goods": "Consumer & Lifestyle",
    "mass_merchants_department_stores": "Consumer & Lifestyle",
    "restaurants": "Consumer & Lifestyle",
    "shopping": "Consumer & Lifestyle",
    "skin_nail_care": "Consumer & Lifestyle",
    "spas_beauty_services": "Consumer & Lifestyle",
    "swimming_pools_spas": "Consumer & Lifestyle",
    "t_birthday": "Consumer & Lifestyle",
    "t_bread": "Consumer & Lifestyle",
    "t_coffee": "Consumer & Lifestyle",
    "t_eggs": "Consumer & Lifestyle",
    "t_grocery_store": "Consumer & Lifestyle",
    "t_meat": "Consumer & Lifestyle",
    "t_milk": "Consumer & Lifestyle",
    "tobacco_products": "Consumer & Lifestyle",
    "weddings": "Consumer & Lifestyle",

    # ---------- Energy ----------
    "t_natural_gas": "Energy",
    "t_petroleum": "Energy",
    "electricity": "Energy",
    "energy_utilities": "Energy",
    "fuel_economy_gas_prices": "Energy",
    "oil_gas": "Energy",
    "renewable_alt_energy": "Energy",
    "t_electricity": "Energy",
    "t_gasoline": "Energy",
    "water_supply_treatment": "Energy",

    # ---------- Health ----------
    "health": "Health",
    "health_insurance": "Health",
    "hospitals_treatment_centers": "Health",
    "medical_facilities_services": "Health",
    "mental_health": "Health",
    "pharmacy": "Health",
    "veterinarians": "Health",
    "pharmaceuticals_biotech": "Health",

    # ---------- Public Sector ----------
    "politics": "Public Sector",
    "government": "Public Sector",
    "education": "Public Sector",
    "social_issues_advocacy": "Public Sector",
    "charity_philanthropy": "Public Sector",
    "emergency_services": "Public Sector",
    "environmental_issues": "Public Sector",
    "ethnic_identity_groups": "Public Sector",
    "law_government": "Public Sector",
    "social_services": "Public Sector",
    "waste_management": "Public Sector",

    # ---------- Other ----------
    "offbeat": "Other",
    "online_goodies": "Other",
    "people_society": "Other",
    "religion_belief": "Other",
}

# ---------------------------------------------------------------------
# SECTOR COLORS
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# SECTOR COLORS (Tableau-style pastel palette)
# ---------------------------------------------------------------------
SECTOR_COLORS = {
    # Primary hues
    "Macro & Finance":      "#4E79A7",   # blue
    "Real Estate":          "#E15759",   # red
    "Tech & Media":         "#EDC949",   # yellow
    "Business & Industry":  "#59A14F",   # green

    # Secondary hues
    "Travel & Tourism":     "#76B7B2",   # teal
    "Consumer & Lifestyle": "#FF9DA7",   # pink
    "Labour & Income":      "#F28E2C",   # orange
    "Health":               "#C39BD3",   # lavender

    # Tertiary hues
    "Public Sector":        "#6C5FA4",   # indigo
    "Energy":               "#9C755F",   # brown
    "Other":                "#BAB0AC",   # warm grey
}

plt.rcParams.update({
    "figure.facecolor": NEUTRAL_FIG,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": "#C8C8C8",
    "axes.labelcolor": NEUTRAL_TEXT,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": NEUTRAL_TEXT,
    "grid.color": "#FFFFFF",
    "grid.alpha": 0.85,
    "grid.linewidth": 1.0,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
})

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def save_figure(fig, png_path, pdf_path, extra_artists=None):
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=extra_artists,
        pad_inches=SAVE_PAD,
        facecolor=fig.get_facecolor()
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        bbox_extra_artists=extra_artists,
        pad_inches=SAVE_PAD,
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)


def get_base_variable(name):
    match = re.match(r"^(.+)_w([1-4])$", str(name))
    return match.group(1) if match else name


def is_exogenous(name):
    return not str(name).startswith("C_") and not str(name).startswith("infl_yoy_lag")


def assign_sector(base_variable):
    return SECTOR_MAPPING.get(base_variable, "Other")


def assign_period(date):
    date = pd.Timestamp(date)

    for name, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return name

    return "Other"


def prepare_sector_data():
    if not PERM_INPUT_CSV.exists():
        print(f"[ERROR] File not found: {PERM_INPUT_CSV}")
        return None, None

    df = pd.read_csv(PERM_INPUT_CSV)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        if "period" not in df.columns:
            df["period"] = df["date"].apply(assign_period)

    if "base_variable" not in df.columns:
        if "feature" not in df.columns:
            print("[ERROR] Neither 'base_variable' nor 'feature' found.")
            return None, None

        df["base_variable"] = df["feature"].apply(get_base_variable)

    if "mean_rmse_increase" in df.columns:
        value_col = "mean_rmse_increase"
    elif "total_rmse_increase" in df.columns:
        value_col = "total_rmse_increase"
    else:
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        if not numeric_cols:
            print("[ERROR] No numeric importance column found.")
            return None, None

        value_col = numeric_cols[0]

    df = df[df["base_variable"].apply(is_exogenous)].copy()

    feature_means = (
        df.groupby(["period", "base_variable"], as_index=False)[value_col]
        .mean()
    )

    if CLIP_NEGATIVE_IMPORTANCE:
        feature_means[value_col] = feature_means[value_col].clip(lower=0)

    feature_means["sector"] = feature_means["base_variable"].apply(assign_sector)

    sector_totals = (
        feature_means.groupby(["period", "sector"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "sector_total"})
    )

    periods_list = [
        p for p in PERIODS_ORDER
        if p in sector_totals["period"].values
    ]

    if not periods_list:
        print("[ERROR] No valid periods found.")
        return None, None

    return sector_totals, periods_list


# ---------------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------------
def plot_sector_rank_evolution():
    sector_totals, periods_list = prepare_sector_data()

    if sector_totals is None:
        return

    pivot = (
        sector_totals
        .pivot(index="sector", columns="period", values="sector_total")
        .fillna(0.0)
        .reindex(columns=periods_list)
    )

    pivot = pivot[pivot.sum(axis=1) > 1e-6]

    ranks = pivot.rank(axis=0, ascending=False, method="min").astype(int)

    fig, ax = plt.subplots(
        figsize=(9.4, 0.42 * len(ranks) + 1.8),
        facecolor=NEUTRAL_FIG
    )

    # More white space on the right for labels outside the plot border
    fig.subplots_adjust(right=RIGHT_MARGIN)

    x_positions = np.arange(len(periods_list))
    label_artists = []

    for sector, row in ranks.iterrows():
        color = SECTOR_COLORS.get(sector, "#BFC3C8")
        y_vals = row.values

        ax.plot(
            x_positions,
            y_vals,
            color=color,
            linewidth=2.4,
            alpha=0.95,
            zorder=3
        )

        ax.scatter(
            x_positions,
            y_vals,
            color=color,
            s=75,
            alpha=1.0,
            zorder=4,
            edgecolor="white",
            linewidth=1.3
        )

        txt = ax.text(
            x_positions[-1] + LABEL_X_OFFSET,
            y_vals[-1],
            sector,
            va="center",
            ha="left",
            fontsize=9,
            color=color,
            fontweight="bold",
            clip_on=False
        )

        label_artists.append(txt)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(periods_list, fontsize=11)

    ax.set_ylabel("Rank", fontsize=11)

    # Show all ranks: 1, 2, ..., 10
    ax.set_yticks(np.arange(1, len(ranks) + 1, 1))
    ax.set_yticklabels(np.arange(1, len(ranks) + 1, 1))

    ax.invert_yaxis()

    # Equal distance:
    # left border to first points = right border to last points
    ax.set_xlim(
        x_positions[0] - POINT_MARGIN,
        x_positions[-1] + POINT_MARGIN
    )

    ax.grid(axis="y", alpha=0.85)
    ax.set_axisbelow(True)

    # Create full border: left, bottom, top, right
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#C8C8C8")
        ax.spines[spine].set_linewidth(1.0)

    save_figure(
        fig,
        GRAPH72_PNG,
        GRAPH72_PDF,
        extra_artists=label_artists
    )

    print(f"Saved: {GRAPH72_PNG}")
    print(f"Saved: {GRAPH72_PDF}")


if __name__ == "__main__":
    OUTPUT_GRAPHS.mkdir(parents=True, exist_ok=True)
    plot_sector_rank_evolution()
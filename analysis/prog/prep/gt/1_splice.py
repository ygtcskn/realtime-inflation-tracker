import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import interpolate

from config import DATA_RAW, DATA_TEMP

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

RAW_MONTHLY_DIR = DATA_RAW / "raw_monthly"
RAW_WEEKLY_DIR = DATA_RAW / "raw_weekly"
OUT_DIR = DATA_TEMP

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "TR", "GB", "US"]

CAT_MAP = {
    "3": "arts_entertainment", "5": "computers_electronics", "7": "finance",
    "8": "games", "11": "home_garden", "12": "business_industrial",
    "13": "internet_telecom", "14": "people_society", "16": "news",
    "18": "shopping", "19": "law_government", "20": "sports",
    "22": "books_literature", "23": "performing_arts", "24": "visual_art_design",
    "25": "advertising_marketing", "28": "office_services", "29": "real_estate",
    "30": "computer_hardware", "31": "programming", "32": "software",
    "33": "offbeat", "34": "movies", "35": "music_audio", "36": "tv_video",
    "37": "banking", "38": "insurance", "39": "card_games",
    "41": "computer_video_games", "43": "online_goodies", "44": "beauty_fitness",
    "45": "health", "46": "agriculture_forestry", "47": "autos_vehicles",
    "48": "construction_maintenance", "49": "manufacturing",
    "50": "transportation_logistics", "53": "web_hosting_domain_registration",
    "54": "social_issues_advocacy", "55": "dating_personals",
    "56": "ethnic_identity_groups", "57": "charity_philanthropy",
    "59": "religion_belief", "60": "jobs", "64": "antiques_collectibles",
    "65": "hobbies_leisure", "66": "pets_animals", "67": "travel",
    "68": "apparel", "69": "consumer_resources", "70": "gifts_special_event_items",
    "71": "food_drink", "73": "mass_merchants_department_stores", "74": "education",
    "75": "legal", "76": "government", "77": "enterprise_technology",
    "78": "consumer_electronics", "82": "environmental_issues",
    "83": "marketing_services", "84": "search_engine_optimization_marketing",
    "89": "vehicle_parts_accessories", "93": "skin_nail_care", "94": "fitness",
    "95": "office_supplies", "96": "real_estate_agencies",
    "97": "consumer_advocacy_protection", "99": "gifts",
    "121": "grocery_food_retailers", "123": "tobacco_products",
    "144": "unwanted_body_facial_hair_removal", "145": "spas_beauty_services",
    "158": "home_improvement", "168": "emergency_services",
    "170": "vehicle_licensing_registration", "179": "hotels_accommodations",
    "205": "car_rental_taxi_services", "206": "cruises_charters",
    "208": "tourist_destinations", "233": "energy_utilities", "248": "pharmacy",
    "249": "health_insurance", "250": "hospitals_treatment_centers",
    "255": "pharmaceuticals_biotech", "256": "medical_facilities_services",
    "270": "home_furnishings", "271": "home_appliances", "276": "restaurants",
    "277": "alcoholic_beverages", "278": "accounting_auditing",
    "279": "credit_lending", "287": "industrial_materials_equipment",
    "288": "chemicals_industry", "289": "freight_trucking", "290": "packaging",
    "291": "moving_relocation", "293": "weddings", "314": "computer_security",
    "329": "business_services", "334": "corporate_events",
    "341": "customer_relationship_management_crm",
    "342": "enterprise_resource_planning_erp", "343": "data_management",
    "354": "import_export", "355": "book_retailers", "365": "coupons_discount_offers",
    "378": "apartments_residential_rentals", "380": "veterinarians", "396": "politics",
    "408": "newspapers", "423": "bankruptcy", "425": "property_management",
    "437": "mental_health", "465": "home_insurance", "466": "home_financing",
    "468": "auto_financing", "473": "vehicle_shopping", "477": "architecture",
    "508": "social_services", "566": "textiles_nonwovens", "569": "events_listings",
    "606": "metals_mining", "610": "trucks_suvs", "612": "entertainment_industry",
    "621": "food_production", "650": "building_materials_supplies",
    "651": "civil_engineering", "652": "construction_consulting_contracting",
    "657": "renewable_alt_energy", "658": "electricity", "659": "oil_gas",
    "660": "waste_management", "662": "aviation", "664": "distribution_logistics",
    "665": "maritime_transport", "666": "rail_transport", "667": "urban_transport",
    "670": "agrochemicals", "672": "coatings_adhesives", "673": "dyes_pigments",
    "696": "luxury_goods", "697": "footwear", "718": "outsourcing",
    "728": "computer_servers", "730": "development_tools", "747": "aquaculture",
    "748": "agricultural_equipment", "750": "forestry", "784": "business_news",
    "794": "gps_navigation", "802": "developer_jobs", "813": "college_financing",
    "815": "vehicle_brands", "832": "flooring", "841": "retail_trade",
    "882": "animal_products_services", "894": "acting_theater", "918": "fast_food",
    "952": "swimming_pools_spas", "960": "job_listings", "961": "resumes_portfolios",
    "969": "legal_services", "1003": "luggage_travel_accessories",
    "1010": "travel_agencies_services", "1080": "real_estate_listings",
    "1081": "timeshares_vacation_properties", "1140": "boats_watercraft",
    "1143": "entertainment_media", "1159": "business_operations",
    "1160": "commercial_lending", "1162": "consulting", "1164": "economy_news",
    "1176": "printing_publishing", "1188": "car_electronics",
    "1199": "professional_trade_associations", "1209": "world_news",
    "1214": "commercial_vehicles", "1268": "fuel_economy_gas_prices",
    "1269": "vehicle_fuels_lubricants", "1300": "cad_cam", "1306": "parking",
    "1339": "carpooling_ridesharing", "1349": "water_supply_treatment",
    "10001": "t_inflation", "10002": "t_cost_of_living",
    "10003": "t_consumer_price_index", "10004": "t_price",
    "10005": "t_purchasing_power", "10006": "t_cheap",
    "10007": "t_interest_rate", "10008": "t_central_bank",
    "10009": "t_monetary_policy", "10010": "t_exchange_rate",
    "10011": "t_currency", "10012": "t_foreign_exchange_market",
    "10013": "t_rent", "10014": "t_real_estate", "10015": "t_housing",
    "10016": "t_electricity", "10017": "t_natural_gas", "10018": "t_gasoline",
    "10019": "t_petroleum", "10020": "t_invoice", "10021": "t_public_transport",
    "10022": "t_auto_insurance", "10023": "t_grocery_store", "10024": "t_bread",
    "10025": "t_milk", "10026": "t_eggs", "10027": "t_meat", "10028": "t_coffee",
    "10029": "t_restaurant", "10030": "t_wage", "10031": "t_salary",
    "10032": "t_minimum_wage", "10033": "t_strike_action", "10034": "t_unemployment",
    "10036": "t_savings", "10038": "t_debt", "10039": "t_birthday",
    "10041": "t_unemployment_benefits", "10042": "t_recruitment",
    "10043": "t_investment", "10044": "t_lawyer", "10045": "t_jobs",
    "10046": "t_economic_crisis", "10047": "t_financial_crisis",
    "10048": "t_house_price_index", "10049": "t_crisis", "10050": "t_interest",
    "10051": "t_student_loan", "10052": "t_affordable_housing",
    "10053": "t_recession", "10054": "t_bankruptcy", "10055": "t_export",
    "10056": "t_baggage", "10057": "t_liquidation",
}


def parse_filename(path):
    match = re.match(r"^(\d+)_([A-Z]{2})_[mw]_", os.path.basename(path))
    return (match.group(1), match.group(2)) if match else None


def read_gt_csv(path, freq="M"):
    try:
        df = pd.read_csv(path, skiprows=2, header=None, names=["Date", "value"])
        fmt = "%Y-%m" if freq == "M" else "%Y-%m-%d"
        df["Date"] = pd.to_datetime(df["Date"], format=fmt, errors="coerce")
        df["value"] = pd.to_numeric(df["value"].replace("<1", 0.5), errors="coerce")
        return df.dropna().set_index("Date")["value"].sort_index()

    except Exception:
        return pd.Series(dtype=float)


def interp_m2w(monthly, weekly_dates):
    monthly = monthly.dropna()

    if len(monthly) < 2:
        return pd.Series(index=weekly_dates, dtype=float)

    monthly_num = monthly.index.astype(np.int64) / 1e9 / 86400
    weekly_num = pd.DatetimeIndex(weekly_dates).astype(np.int64) / 1e9 / 86400

    func = interpolate.interp1d(
        monthly_num,
        monthly.values,
        kind="linear",
        fill_value="extrapolate"
    )

    return pd.Series(func(weekly_num), index=weekly_dates)


def splice_window(weekly, monthly):
    start = max(weekly.index.min(), monthly.index.min())
    end = min(weekly.index.max(), monthly.index.max())

    weekly_overlap = weekly[(weekly.index >= start) & (weekly.index <= end)]

    if len(weekly_overlap) == 0 or (weekly_overlap == 0).any():
        return weekly

    monthly_interp = interp_m2w(monthly, weekly_overlap.index)
    ratio = (monthly_interp / weekly_overlap).mean()

    return weekly * (1 if np.isnan(ratio) else ratio)


def run_splicing():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_data = {}

    monthly_files = glob.glob(str(RAW_MONTHLY_DIR / "*.csv"))

    for path in tqdm(monthly_files, desc="Monthly GT files", unit="file"):
        meta = parse_filename(path)

        if not meta or meta[1] not in COUNTRIES or meta[0] not in CAT_MAP:
            continue

        series = read_gt_csv(path, "M")

        if not series.empty:
            monthly_data[f"{meta[1]}_{CAT_MAP[meta[0]]}"] = series

    df_monthly = pd.DataFrame(monthly_data).sort_index()
    df_monthly.to_csv(OUT_DIR / "1_monthly_raw.csv")

    series_buffer = {}

    weekly_files = glob.glob(str(RAW_WEEKLY_DIR / "*.csv"))

    for path in tqdm(weekly_files, desc="Weekly GT files", unit="file"):
        meta = parse_filename(path)

        if not meta or meta[1] not in COUNTRIES or meta[0] not in CAT_MAP:
            continue

        series = read_gt_csv(path, "W")

        if not series.empty:
            col = f"{meta[1]}_{CAT_MAP[meta[0]]}"
            series_buffer.setdefault(col, []).append(series)

    spliced_data = {}

    for col, chunks in tqdm(series_buffer.items(), desc="Splicing series", unit="series"):
        if col not in df_monthly.columns:
            continue

        adjusted_chunks = [splice_window(chunk, df_monthly[col]) for chunk in chunks]
        spliced_data[col] = pd.concat(adjusted_chunks).groupby(level=0).mean().sort_index()

    df_spliced = pd.DataFrame(spliced_data).sort_index()
    df_spliced.to_csv(OUT_DIR / "1_weekly_spliced_raw.csv")

    return df_spliced


if __name__ == "__main__":
    run_splicing()
    print(f"Done. Spliced files saved to: {OUT_DIR}")
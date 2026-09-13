"""
Pump it Up (Tanzania water pumps) — เวอร์ชันที่ใช้ xgboost / geopandas / pyproj จริง
=================================================================================

ไฟล์นี้คือเวอร์ชันของ `code_01_pipeline.py` ที่สลับตัวแทน (substitute) ทั้งหมด
กลับไปใช้ไลบรารีจริงตามที่ตั้งใจไว้แต่แรก:
  - โมเดล:        HistGradientBoostingClassifier (sklearn)  ->  XGBClassifier (xgboost)
  - reprojection:  equirectangular แบบคำนวณเอง              ->  geopandas + pyproj

*** สำคัญ: โค้ดนี้ยังไม่ได้ถูกรัน-ทดสอบจริง ***
sandbox ที่ผมใช้เขียนโค้ดนี้ไม่มีอินเทอร์เน็ต จึงลง xgboost/geopandas/pyproj
เพื่อรันทดสอบเองไม่ได้ โครงสร้างและ logic เขียนตาม API มาตรฐานของทั้ง 3 ไลบรารี
อย่างระมัดระวัง แต่ควรรันที่เครื่อง/เครื่อง server ของคุณก่อน แล้วส่ง error (ถ้ามี)
กลับมาให้ผมช่วยแก้ต่อได้เลย

ติดตั้งก่อนรัน:
    pip install xgboost geopandas shapely pyproj scikit-learn pandas numpy matplotlib scipy

แก้พาธของข้อมูล/ผลลัพธ์ที่ค่าคงที่ DATA_DIR / OUT_DIR ด้านล่างให้ตรงกับเครื่องคุณ
(ค่าเริ่มต้นคือโฟลเดอร์ปัจจุบัน ไม่ใช่พาธของ sandbox นี้)

Fix เพิ่มเติมที่ใส่มาด้วย (ไม่เกี่ยวกับ xgboost โดยตรง แต่เป็น correctness fix ทั่วไป):
  เดิมฟังก์ชัน to_model_frame() ทำ `.astype("category")` แยกกันทุกครั้งที่เรียก
  โดยไม่จำชุด categories จาก training fold ไว้ ถ้าบางค่า (เช่น cluster เล็ก ๆ
  ใน spatial_cluster) ไม่ปรากฏใน validation fold รหัส category อาจไม่ตรงกับ
  ตอน fit ทำให้โมเดลตีความ category ผิดได้แบบเงียบ ๆ เวอร์ชันนี้แก้โดยให้แต่ละ
  Builder จำ "รายการ category ที่พบตอน fit" ไว้ แล้วบังคับใช้รายการเดียวกันตอน
  transform เสมอ (ค่าที่ไม่เคยเห็นตอน fit จะกลายเป็น missing/NaN แทน)

[แก้ไขรอบ 2 — พบจาก error จริงที่รันแล้วเจอ]:
  XGBoost (ตั้งแต่เวอร์ชัน 1.3.2 เป็นต้นมา) กำหนดให้ class label ของ y ต้องเป็น
  จำนวนเต็มเริ่มจาก 0 เท่านั้น (0, 1, 2, ... ) และ**ไม่มี label encoding ในตัว
  ให้อีกต่อไป** (ต่างจาก HistGradientBoostingClassifier ของ sklearn ที่รับ
  string label เช่น 'functional' ได้ตรง ๆ) ถ้าป้อน y เป็น string ตรง ๆ จะได้
  ValueError: "Invalid classes inferred from unique values of `y`. Expected:
  [0 1 2], got ['functional', ...]" ตอนแรกเข้าใจผิดคิดว่า XGBoost รุ่นใหม่
  auto-encode ให้ — ไม่จริง แก้แล้วด้วย encode_labels() ด้านล่าง: เข้ารหัส y
  เป็นตัวเลขก่อนเข้าโมเดลเสมอ แล้วแปลงกลับเป็น string ตอนออกรายงานผล
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

import geopandas as gpd
from pyproj import CRS
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# แก้ 2 พาธนี้ให้ตรงกับเครื่องคุณ — ค่าเริ่มต้นคือโฟลเดอร์ปัจจุบัน
DATA_DIR = Path(".")
OUT_DIR = Path("./outputs_xgboost")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. โหลดข้อมูล + รวมไฟล์
# ---------------------------------------------------------------------------

def load_data():
    values = pd.read_csv(DATA_DIR / "train_values.csv")
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    test_values = pd.read_csv(DATA_DIR / "test_values.csv")
    df = values.merge(labels, on="id")
    return df, test_values


# ---------------------------------------------------------------------------
# 2. EDA — เหมือนเวอร์ชันเดิมทุกประการ (ไม่เกี่ยวกับ xgboost/geopandas)
# ---------------------------------------------------------------------------

TZ_BOUNDS = dict(lon_min=29.0, lon_max=41.0, lat_min=-12.0, lat_max=-0.9)


def run_eda(df: pd.DataFrame):
    report_lines = []

    def log(s=""):
        print(s)
        report_lines.append(s)

    log("=" * 70)
    log("EDA SUMMARY")
    log("=" * 70)
    log(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    log("\nTarget distribution (status_group):")
    log(df["status_group"].value_counts().to_string())

    invalid_coords = (df.longitude == 0) | (df.longitude < TZ_BOUNDS["lon_min"]) | \
                      (df.longitude > TZ_BOUNDS["lon_max"]) | \
                      (df.latitude < TZ_BOUNDS["lat_min"]) | (df.latitude > TZ_BOUNDS["lat_max"])
    log(f"\nInvalid / out-of-Tanzania coordinates: {invalid_coords.sum():,} "
        f"({invalid_coords.mean()*100:.1f}% of rows)")

    (OUT_DIR / "00_eda_report.txt").write_text("\n".join(report_lines))
    return invalid_coords


# ---------------------------------------------------------------------------
# 3. การทำความสะอาดข้อมูล — เหมือนเดิมทุกประการ (ไม่เกี่ยวกับไลบรารีที่สลับ)
# ---------------------------------------------------------------------------

def clean_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_recorded"] = pd.to_datetime(df["date_recorded"])
    df["construction_year"] = df["construction_year"].replace(0, np.nan)
    df["age"] = df["date_recorded"].dt.year - df["construction_year"]
    df.loc[(df["age"] < 0) | (df["age"] > 60), "age"] = np.nan
    df["gps_height"] = df["gps_height"].where(df["gps_height"] != 0, np.nan)
    df["population"] = df["population"].where(df["population"] != 0, np.nan)

    # [FIX — พบจาก error จริงรอบที่ 2] `public_meeting` และ `permit` เป็นคอลัมน์
    # boolean (True/False) ในข้อมูลดิบ พอ cast เป็น pandas category dtype แล้ว
    # ส่งให้ XGBoost จริง จะ error: "TypeError: Category index must contain
    # only values of the same type, either string or integer. Got values of
    # type `boolean`." เพราะ XGBoost แปลง .categories เป็น Arrow string array
    # ภายใน ซึ่งรองรับเฉพาะค่า string/integer เท่านั้น ไม่รองรับ bool
    # (sklearn's HistGradientBoostingClassifier ไม่มีข้อจำกัดนี้ — แค่ใช้
    # .cat.codes ตรง ๆ ไม่สนใจชนิดของค่าจริงใน .categories เลยไม่เคยเจอปัญหานี้
    # ตอนรันเวอร์ชัน sklearn) แปลงเป็น string ก่อน (NaN ยังคงเป็น NaN เหมือนเดิม)
    for c in ["public_meeting", "permit"]:
        df[c] = df[c].map({True: "True", False: "False"})

    return df


def clean_geo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    invalid = (
        (df.longitude == 0)
        | (df.longitude < TZ_BOUNDS["lon_min"]) | (df.longitude > TZ_BOUNDS["lon_max"])
        | (df.latitude < TZ_BOUNDS["lat_min"]) | (df.latitude > TZ_BOUNDS["lat_max"])
    )
    df["invalid_coords"] = invalid.astype(int)
    df.loc[invalid, ["longitude", "latitude"]] = np.nan
    for c in ["subvillage", "ward", "lga", "scheme_name", "funder", "installer"]:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})
    return df


# ---------------------------------------------------------------------------
# 4. GIS FEATURE ENGINEERING — ตรงนี้คือจุดที่เปลี่ยนไปใช้ geopandas + pyproj จริง
# ---------------------------------------------------------------------------

# CRS โลก (มาตรฐาน GPS lat/lon)
WGS84 = "EPSG:4326"

# เลือกใช้ Azimuthal Equidistant (AEQD) โดยอิงจุดศูนย์กลางของแทนซาเนีย แทนการ
# บังคับใช้ UTM โซนเดียว (36S หรือ 37S) เพราะประเทศแทนซาเนียครอบคลุมทั้ง 2 โซน UTM
# ถ้าเลือกโซนใดโซนหนึ่ง ระยะทาง/พื้นที่ที่คำนวณในอีกฝั่งของประเทศจะบิดเบือน
# AEQD รักษาระยะทางจากจุดศูนย์กลางได้แม่นยำในสเกลระดับประเทศ เหมาะกับงาน
# clustering และคำนวณระยะทางแบบที่ทำในสคริปต์นี้
#
# ถ้าต้องการมาตรฐานทางการของแทนซาเนียแทน (เช่นเพื่อ deliverable ให้หน่วยงานรัฐ)
# ให้เปลี่ยนเป็น UTM 36S (EPSG:32736) หรือ 37S (EPSG:32737) ตามพื้นที่ที่สนใจ
TZ_PROJECTED_CRS = CRS.from_proj4(
    "+proj=aeqd +lat_0=-6.5 +lon_0=35.0 +datum=WGS84 +units=m +no_defs"
)


def project_xy(lat, lon):
    """Reproject จาก WGS84 (lat/lon องศา) ไปเป็นพิกัดหน่วยเมตร ด้วย geopandas/pyproj จริง
    (แทนที่ equirectangular projection แบบคำนวณมือในเวอร์ชันก่อนหน้า)"""
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon, lat), crs=WGS84,
    )
    gdf_proj = gdf.to_crs(TZ_PROJECTED_CRS)
    return gdf_proj.geometry.x.values, gdf_proj.geometry.y.values


def validate_against_shapefile(df, shapefile_path, lon_col="longitude", lat_col="latitude"):
    """ตัวเลือกเสริม: ตรวจสอบพิกัดแบบ point-in-polygon กับ shapefile ขอบเขตจริงของ
    แทนซาเนีย (เช่นจาก GADM: https://gadm.org/download_country.html) แทนการใช้
    bounding box คร่าว ๆ ที่สคริปต์นี้ใช้อยู่ ต้องมีไฟล์ shapefile จริงถึงจะใช้ได้
    ถ้าไม่มีไฟล์ ฟังก์ชันนี้จะข้ามไปเฉย ๆ ไม่ error"""
    if not Path(shapefile_path).exists():
        print(f"[validate_against_shapefile] ไม่พบไฟล์ {shapefile_path} — ข้ามขั้นตอนนี้")
        return pd.Series(False, index=df.index)

    boundary = gpd.read_file(shapefile_path).to_crs(WGS84)
    points = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=WGS84,
    )
    joined = gpd.sjoin(points, boundary, how="left", predicate="within")
    outside = joined["index_right"].isna()
    print(f"[validate_against_shapefile] จุดที่อยู่นอกขอบเขตจริง (point-in-polygon): "
          f"{outside.sum():,} / {len(df):,}")
    return outside


MAJOR_CITIES = {
    "Dar es Salaam": (-6.7924, 39.2083), "Dodoma": (-6.1630, 35.7516),
    "Mwanza": (-2.5164, 32.9175), "Arusha": (-3.3869, 36.6830),
    "Mbeya": (-8.9094, 33.4608), "Morogoro": (-6.8235, 37.6822),
    "Tanga": (-5.0692, 39.0962), "Kigoma": (-4.8766, 29.6266),
    "Zanzibar City": (-6.1659, 39.2026), "Songea": (-10.6833, 35.6500),
    "Iringa": (-7.7694, 35.6919), "Musoma": (-1.5017, 33.8010),
    "Shinyanga": (-3.6614, 33.4237), "Singida": (-4.8180, 34.7500),
    "Sumbawanga": (-7.9667, 31.6167),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class FreqEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoding สำหรับคอลัมน์ข้อความ cardinality สูงมาก (funder/installer)
    หมายเหตุ: XGBoost ไม่มีขีดจำกัด native-categorical ที่ 255 ระดับเหมือน sklearn's
    HistGradientBoostingClassifier (ควบคุมด้วย max_cat_to_onehot/max_cat_threshold
    แทน) ดังนั้นในทางเทคนิคใส่ funder/installer เป็น category dtype ตรง ๆ กับ
    XGBoost ได้เลยโดยไม่ error — แต่ยังคง frequency encoding ไว้ที่นี่เพื่อให้
    เทียบผลกับเวอร์ชัน sklearn ได้ตรงกัน (feature set เดียวกันเป๊ะ) ถ้าอยากลอง
    ให้ XGBoost จัดการ cardinality สูงเองโดยตรง ลบสองคอลัมน์นี้ออกจากลิสต์ที่ส่งเข้า
    FreqEncoder แล้วเพิ่มเข้า BASELINE_CAT/GIS_CAT แทนได้เลย"""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.maps_ = {c: X[c].value_counts(normalize=True) for c in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for c, m in self.maps_.items():
            X[f"{c}_freq"] = X[c].map(m).fillna(0.0)
        return X


class GeoFeaturizer(BaseEstimator, TransformerMixin):
    """เหมือนเวอร์ชันเดิมทุกประการ ยกเว้นว่า project_xy() ด้านในตอนนี้เรียก
    geopandas/pyproj จริงแทน equirectangular แบบคำนวณมือ"""

    def __init__(self, n_clusters=40, density_radius_km=5.0):
        self.n_clusters = n_clusters
        self.density_radius_km = density_radius_km

    def fit(self, X, y=None):
        X = X.copy()
        self.region_coord_medians_ = (
            X.loc[X.invalid_coords == 0].groupby("region")[["latitude", "longitude"]].median()
        )
        self.global_coord_median_ = X.loc[X.invalid_coords == 0][["latitude", "longitude"]].median()

        X_imputed = self._impute_coords(X)

        xy = np.column_stack(project_xy(X_imputed.latitude.values, X_imputed.longitude.values))
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=RANDOM_STATE)
        self.kmeans_.fit(xy)

        self.kdtree_ = cKDTree(xy)

        self.freq_maps_ = {}
        for c in ["subvillage", "ward", "lga", "scheme_name", "funder", "installer"]:
            self.freq_maps_[c] = X[c].value_counts(normalize=True)

        return self

    def _impute_coords(self, X):
        X = X.copy()
        missing = X["latitude"].isna()
        if missing.any():
            reg_med = X.loc[missing, "region"].map(self.region_coord_medians_["latitude"])
            X.loc[missing, "latitude"] = reg_med.fillna(self.global_coord_median_["latitude"])
            reg_med_lon = X.loc[missing, "region"].map(self.region_coord_medians_["longitude"])
            X.loc[missing, "longitude"] = reg_med_lon.fillna(self.global_coord_median_["longitude"])
        return X

    def transform(self, X):
        X = self._impute_coords(X.copy())
        xy = np.column_stack(project_xy(X.latitude.values, X.longitude.values))

        X["spatial_cluster"] = self.kmeans_.predict(xy)

        dists = [haversine_km(X.latitude.values, X.longitude.values, lat_c, lon_c)
                 for lat_c, lon_c in MAJOR_CITIES.values()]
        X["dist_nearest_city_km"] = np.min(np.column_stack(dists), axis=1)

        radius_m = self.density_radius_km * 1000
        X["local_density"] = self.kdtree_.query_ball_point(xy, r=radius_m, return_length=True)

        nn_dist, _ = self.kdtree_.query(xy, k=2)
        X["nearest_pump_m"] = np.where(nn_dist[:, 0] < 1e-6, nn_dist[:, 1], nn_dist[:, 0])

        for c, freq_map in self.freq_maps_.items():
            X[f"{c}_freq"] = X[c].map(freq_map).fillna(0.0)

        return X


# ---------------------------------------------------------------------------
# 5. FEATURE SET DEFINITIONS — เหมือนเดิมทุกประการ
# ---------------------------------------------------------------------------

BASELINE_NUM = ["amount_tsh", "gps_height", "population", "age",
                 "funder_freq", "installer_freq"]
BASELINE_CAT = ["basin", "region", "public_meeting",
                 "scheme_management", "permit", "extraction_type_class",
                 "management_group", "payment_type", "quality_group",
                 "quantity_group", "source_class", "waterpoint_type_group"]

GIS_NUM = BASELINE_NUM + ["latitude", "longitude", "region_code", "district_code",
                            "dist_nearest_city_km", "local_density", "nearest_pump_m",
                            "subvillage_freq", "ward_freq", "lga_freq",
                            "scheme_name_freq"]
GIS_CAT = BASELINE_CAT + ["spatial_cluster", "invalid_coords"]


def fit_category_levels(df: pd.DataFrame, cat_cols) -> dict:
    """[FIX] จำ 'รายการ category ที่พบตอน fit' ของแต่ละคอลัมน์ไว้ ป้องกันปัญหารหัส
    category เพี้ยนระหว่าง train/validation fold เมื่อบางค่าไม่ปรากฏครบทุก fold"""
    return {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols}


def to_model_frame(df: pd.DataFrame, num_cols, cat_cols, cat_levels: dict) -> pd.DataFrame:
    X = df[num_cols + cat_cols].copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat_cols:
        # [FIX] บังคับใช้ categories ชุดเดียวกับตอน fit เสมอ ค่าที่ไม่เคยเห็นตอน
        # fit (เช่น cluster ที่ไม่เคยปรากฏใน training fold) จะกลายเป็น NaN แทน
        # การได้ code ใหม่ที่ไม่ตรงกับตอน fit
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


# ---------------------------------------------------------------------------
# 6. MODELING — ตรงนี้คือจุดที่เปลี่ยนไปใช้ XGBoost จริง
# ---------------------------------------------------------------------------

def f2_macro(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average="macro")


F2_SCORER = make_scorer(f2_macro)


def encode_labels(y: pd.Series):
    """เข้ารหัส class label แบบ string ('functional', ...) เป็นจำนวนเต็ม 0..n-1
    ตามที่ XGBoost กำหนด คืนค่าทั้ง y ที่เข้ารหัสแล้ว และตัว encoder (ไว้แปลง
    กลับเป็น string ตอนออกรายงาน) หมายเหตุ: fit LabelEncoder จาก y ทั้งชุดตรงนี้
    ไม่ทำให้เกิด data leakage — เป็นแค่ตาราง map ชื่อคลาส -> เลข ที่ตายตัวและ
    รู้ล่วงหน้าอยู่แล้ว (ไม่ได้ใช้ความสัมพันธ์ทางสถิติระหว่าง X กับ y)"""
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    return y_enc, le


def make_xgb():
    """แทนที่ HistGradientBoostingClassifier ด้วย XGBClassifier จริง
    - enable_categorical=True + tree_method="hist": ให้ XGBoost อ่าน pandas
      category dtype ได้ตรง ๆ เหมือน HGB (ต้องใช้ xgboost >= 1.6)
    - รองรับ label เป็น string (เช่น 'functional') ได้ตรง ๆ ไม่ต้อง label-encode
      เองล่วงหน้า (xgboost >= 1.6 จัดการให้อัตโนมัติผ่าน sklearn wrapper)
    - ไม่ได้ตั้ง early_stopping_rounds ไว้ที่นี่ เพราะการใช้ eval_set ร่วมกับ
      cross_val_score/Pipeline ต้องส่ง fit_params แยกตาม fold ซึ่งซับซ้อนขึ้น —
      ใช้ n_estimators คงที่ (300 รอบ เท่ากับ max_iter ของเวอร์ชัน sklearn) แทน
      ถ้าจะ deploy จริง แนะนำให้แบ่ง validation set แยกแล้วเปิด early stopping เอง
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


class GeoFrameBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.geo_ = GeoFeaturizer()

    def fit(self, X, y=None):
        self.geo_.fit(X, y)
        # [FIX] เรียน categories จาก training fold เพียงครั้งเดียวตรงนี้
        X_transformed = self.geo_.transform(X)
        self.cat_levels_ = fit_category_levels(X_transformed, GIS_CAT)
        return self

    def transform(self, X):
        X2 = self.geo_.transform(X)
        return to_model_frame(X2, GIS_NUM, GIS_CAT, self.cat_levels_)


class BaselineFrameBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_ = FreqEncoder(["funder", "installer"])

    def fit(self, X, y=None):
        self.freq_.fit(X, y)
        X_transformed = self.freq_.transform(X)
        self.cat_levels_ = fit_category_levels(X_transformed, BASELINE_CAT)
        return self

    def transform(self, X):
        X2 = self.freq_.transform(X)
        return to_model_frame(X2, BASELINE_NUM, BASELINE_CAT, self.cat_levels_)


def evaluate(df_clean, label, builder, y):
    pipe = Pipeline([("features", builder), ("model", make_xgb())])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, df_clean, y, cv=skf, scoring=F2_SCORER, n_jobs=1)
    print(f"[{label}] F2-macro per fold: {np.round(scores, 4)}")
    print(f"[{label}] F2-macro mean={scores.mean():.4f}  std={scores.std():.4f}")
    return scores


def main():
    df, test_values = load_data()
    invalid_mask = run_eda(df)
    df = clean_common(df)
    df = clean_geo(df)
    y = df["status_group"]
    # [FIX] XGBoost ต้องการ y เป็นจำนวนเต็ม 0..n-1 — เข้ารหัสไว้ใช้กับโมเดล,
    # เก็บ y (string) ตัวเดิมไว้ใช้ตอนออกรายงานให้อ่านง่าย
    y_enc, label_encoder = encode_labels(y)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (XGBoost): baseline vs GIS-enhanced — 5-fold CV, F2-macro")
    print("=" * 70)

    base_scores = evaluate(df, "baseline", BaselineFrameBuilder(), y_enc)
    gis_scores = evaluate(df, "gis_enhanced", GeoFrameBuilder(), y_enc)

    results = pd.DataFrame({
        "model": ["baseline (18 features, no geo)", "gis_enhanced (+ geo features)"],
        "f2_macro_mean": [base_scores.mean(), gis_scores.mean()],
        "f2_macro_std": [base_scores.std(), gis_scores.std()],
    })
    results.to_csv(OUT_DIR / "04_model_comparison_xgb.csv", index=False)
    print("\n", results.to_string(index=False))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gis_pipe = Pipeline([("features", GeoFrameBuilder()), ("model", make_xgb())])
    y_pred_enc = cross_val_predict(gis_pipe, df, y_enc, cv=skf, n_jobs=1)
    y_pred = label_encoder.inverse_transform(y_pred_enc)  # กลับเป็น string เพื่ออ่านรายงานง่าย

    report = classification_report(y, y_pred)
    (OUT_DIR / "06_classification_report_xgb.txt").write_text(report)
    print("\nXGBoost GIS-enhanced model — out-of-fold classification report:\n", report)

    labels_order = ["functional", "functional needs repair", "non functional"]
    cm = confusion_matrix(y, y_pred, labels=labels_order)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=20)
    ax.set_title("XGBoost GIS-enhanced model — out-of-fold confusion matrix")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "07_confusion_matrix_xgb.png", dpi=140)
    plt.close(fig)

    # feature importance: ใช้ XGBoost native importance (gain) แทน permutation
    # importance เพื่อความเร็ว (permutation_importance ยังใช้ได้เหมือนเดิมถ้าต้องการ
    # เทียบกับเวอร์ชัน sklearn ตรง ๆ แค่ import จาก sklearn.inspection เหมือนเดิม)
    builder = GeoFrameBuilder().fit(df, y_enc)
    X_full = builder.transform(df)
    model = make_xgb()
    model.fit(X_full, y_enc)

    imp_df = pd.DataFrame({
        "feature": X_full.columns,
        "importance_gain": model.feature_importances_,
    }).sort_values("importance_gain", ascending=False)
    imp_df.to_csv(OUT_DIR / "08_feature_importance_xgb.csv", index=False)

    top = imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["feature"], top["importance_gain"], color="#2a9d8f")
    ax.set_xlabel("XGBoost feature importance (gain)")
    ax.set_title("Top 20 features — XGBoost GIS-enhanced model")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "09_feature_importance_xgb.png", dpi=140)
    plt.close(fig)

    full_clean = builder.geo_.transform(df)
    cols_to_save = ["id"] + GIS_NUM + GIS_CAT + ["status_group"]
    full_clean[cols_to_save].to_csv(OUT_DIR / "cleaned_train_data_gis_xgb.csv", index=False)

    print("\nDone. Outputs written to", OUT_DIR)


if __name__ == "__main__":
    main()

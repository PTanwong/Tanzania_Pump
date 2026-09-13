# อธิบายโค้ด `code_03_pipeline_xgboost_geopandas.py` แบบละเอียดสำหรับมือใหม่

เอกสารนี้เดินอธิบายไฟล์ `code_03_pipeline_xgboost_geopandas.py` ทีละส่วนตามลำดับ
ที่โค้ดเขียนไว้จริง (comment เลข 1-6 ในไฟล์) พร้อมอธิบายว่า **แต่ละบรรทัดทำอะไร
และทำไมต้องทำแบบนั้น** ไม่ใช่แค่ "โค้ดนี้ทำอะไร" เฉย ๆ

---

## ศัพท์ที่ควรรู้ก่อนอ่าน (Glossary)

| คำศัพท์ | ความหมายง่าย ๆ |
|---|---|
| **DataFrame** | ตารางข้อมูลของ pandas เหมือน Excel sheet ที่เขียนโปรแกรมจัดการได้ |
| **fit / transform** | รูปแบบมาตรฐานของ scikit-learn: `fit()` = "เรียนรู้" อะไรบางอย่างจากข้อมูล (เช่น ค่าเฉลี่ย), `transform()` = "นำสิ่งที่เรียนรู้ไปใช้" กับข้อมูลชุดใหม่ แยก 2 ขั้นตอนนี้ออกจากกันเพื่อป้องกันไม่ให้ "แอบดู" ข้อมูลที่ควรใช้ทดสอบ |
| **Data leakage** | การที่โมเดล "แอบเห็น" ข้อมูลที่ไม่ควรเห็นตอนเทรน ทำให้ผลตอนทดสอบดูดีเกินจริง ทั้งที่ใช้งานจริงจะแย่กว่านั้น |
| **Cross-validation (CV)** | แบ่งข้อมูลเป็นหลายส่วน (เช่น 5 ส่วน) แล้วสลับกันเทรน 4 ส่วน ทดสอบ 1 ส่วน วนจนครบ 5 รอบ เพื่อให้คะแนนที่ได้เชื่อถือได้ ไม่ขึ้นกับการแบ่งข้อมูลแบบใดแบบหนึ่งโดยบังเอิญ |
| **Categorical column** | คอลัมน์ที่เป็น "หมวดหมู่" เช่น ภูมิภาค, ประเภทปั๊ม (ต่างจากตัวเลขที่เอามาบวกลบคูณหารได้จริง) |
| **CRS (Coordinate Reference System)** | ระบบพิกัดที่บอกว่าตัวเลข lat/lon "แปลว่าอะไร" บนโลกจริง — lat/lon องศาปกติ (WGS84) วัดระยะทางตรง ๆ ไม่ได้แม่นยำ ต้อง "แปลง" เป็นหน่วยเมตรก่อน |
| **Class imbalance** | เวลาข้อมูลแต่ละกลุ่ม (class) มีจำนวนไม่เท่ากันมาก เช่นในงานนี้ปั๊ม "functional" มีเยอะกว่า "needs repair" มาก |

---

## ภาพรวม: ไฟล์นี้ทำอะไร

โค้ดทั้งไฟล์คือ pipeline (สายพานการทำงาน) 6 ขั้นตอน ไล่จากบนลงล่าง:

```
1. โหลดข้อมูล  →  2. EDA  →  3. Clean ข้อมูล  →  4. สร้าง GIS feature
                                                            ↓
6. เทรนโมเดล + วัดผล  ←────────────────  5. เลือกว่าจะใช้ feature ไหนเข้าโมเดล
```

เป้าหมายคือเปรียบเทียบว่าโมเดลที่ **ไม่มีข้อมูลตำแหน่ง (geo)** เลย กับโมเดลที่
**มีข้อมูลตำแหน่งเพิ่มเข้าไป** อันไหนทำนายว่าปั๊มน้ำจะเสียได้แม่นกว่ากัน

---

## ส่วนนำเข้าไลบรารี (import) — บรรทัด 41-68

```python
import numpy as np                          # คำนวณตัวเลข/อาเรย์
import pandas as pd                          # จัดการตาราง (DataFrame)
from scipy.spatial import cKDTree            # หาจุดที่อยู่ใกล้กันบนแผนที่ (เร็วมาก)
from sklearn.base import BaseEstimator, TransformerMixin   # แม่แบบสร้าง fit/transform ของตัวเอง
from sklearn.cluster import KMeans           # จัดกลุ่มจุดที่อยู่ใกล้กัน
from sklearn.preprocessing import LabelEncoder  # แปลงชื่อ class เป็นตัวเลข
from sklearn.metrics import ...              # เครื่องมือวัดผลโมเดล (precision, recall, confusion matrix)
from sklearn.model_selection import ...      # เครื่องมือทำ cross-validation
from sklearn.pipeline import Pipeline        # เอาหลายขั้นตอนมาต่อกันเป็นสายพานเดียว

import geopandas as gpd                      # DataFrame ที่รู้จักพิกัดภูมิศาสตร์
from pyproj import CRS                       # กำหนด/แปลงระบบพิกัด
from xgboost import XGBClassifier            # ตัวโมเดลจริงที่ใช้ทำนาย
```

`cKDTree` เด่นตรงที่ถ้ามีปั๊ม 59,400 จุด แล้วอยากรู้ว่า "จุดไหนอยู่ใกล้จุดนี้บ้าง"
ถ้าเทียบทุกคู่ตรง ๆ จะช้ามาก (59,400 × 59,400 ครั้ง) แต่ cKDTree จัดโครงสร้าง
ข้อมูลไว้ล่วงหน้าให้ค้นหาได้เร็วขึ้นมาก

---

## 1. โหลดข้อมูล + รวมไฟล์ (บรรทัด 81-86)

```python
def load_data():
    values = pd.read_csv(DATA_DIR / "train_values.csv")
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    test_values = pd.read_csv(DATA_DIR / "test_values.csv")
    df = values.merge(labels, on="id")
    return df, test_values
```

ข้อมูลต้นฉบับแยกเป็น 2 ไฟล์: `train_values.csv` (รายละเอียดปั๊มแต่ละตัว) กับ
`train_labels.csv` (คำตอบ — ปั๊มตัวนั้น functional/เสีย/ต้องซ่อม) `.merge(on="id")`
คือเอา 2 ตารางมาต่อกันโดยจับคู่ตาม column `id` เหมือน VLOOKUP ใน Excel

---

## 2. EDA (บรรทัด 96-117)

EDA ย่อจาก Exploratory Data Analysis คือ "สำรวจข้อมูลก่อนลงมือทำอะไรจริงจัง"
ฟังก์ชันนี้แค่ **นับ** ปัญหาที่มีในข้อมูล (ไม่ได้แก้อะไร) แล้ว print ออกมาดู + เขียน
ลงไฟล์ `00_eda_report.txt`:

```python
invalid_coords = (df.longitude == 0) | (df.longitude < TZ_BOUNDS["lon_min"]) | ...
```

บรรทัดนี้สร้าง "หน้ากาก" (mask) — อาเรย์ของ True/False ยาวเท่าจำนวนแถว บอกว่า
แถวไหน "พิกัดน่าจะผิด" (อยู่นอกกรอบสี่เหลี่ยมคร่าว ๆ ของประเทศแทนซาเนีย) เช่น
`longitude == 0` เพราะพบว่าบางแถวมีพิกัด (0, 0) ซึ่งจริง ๆ คือจุดกลางมหาสมุทร
ไม่ใช่แทนซาเนีย

---

## 3. การทำความสะอาดข้อมูล (บรรทัด 124-159)

### `clean_common()` — ความสะอาดทั่วไป (ใช้กับทุกโมเดล)

```python
df["date_recorded"] = pd.to_datetime(df["date_recorded"])
df["construction_year"] = df["construction_year"].replace(0, np.nan)
df["age"] = df["date_recorded"].dt.year - df["construction_year"]
df.loc[(df["age"] < 0) | (df["age"] > 60), "age"] = np.nan
```

- `pd.to_datetime` แปลง string วันที่ให้เป็น "วันที่จริง" ที่คำนวณได้ (บวกลบปีได้)
- `construction_year` บางแถวเป็น 0 ซึ่งไม่ใช่ปีจริง (ไม่มีปั๊มไหนสร้างปี ค.ศ. 0)
  แปลว่าเป็นค่าที่ **หายไป** แต่คนบันทึกใส่ 0 แทนช่องว่าง — ต้องแปลงเป็น `NaN`
  (Not a Number = ค่าว่างที่ pandas เข้าใจ) ไม่งั้นโมเดลจะเข้าใจผิดว่า "ปีที่สร้าง
  คือปี 0" จริง ๆ
- สร้างคอลัมน์ `age` (อายุปั๊ม) = ปีที่บันทึก − ปีที่สร้าง แล้วกรองอายุที่เป็นไปไม่ได้
  (ติดลบ หรือมากกว่า 60 ปี) ทิ้งเป็น `NaN` ด้วย

```python
df["gps_height"] = df["gps_height"].where(df["gps_height"] != 0, np.nan)
df["population"] = df["population"].where(df["population"] != 0, np.nan)
```

`.where(condition, other)` แปลว่า "เก็บค่าเดิมไว้ถ้า condition เป็นจริง ไม่งั้น
แทนด้วย other" ดังนั้นบรรทัดนี้แปลว่า "เก็บ gps_height ไว้ถ้าไม่เท่ากับ 0 ถ้าเท่ากับ
0 ให้เปลี่ยนเป็น NaN" — เหตุผลเดียวกับข้างบน: 0 ในคอลัมน์นี้คือ "ไม่มีข้อมูล"
ไม่ใช่ "สูง 0 เมตร" จริง ๆ (ไม่งั้นปั๊มทั้งประเทศจะดูเหมือนอยู่ที่ระดับน้ำทะเลพอดี)

```python
for c in ["public_meeting", "permit"]:
    df[c] = df[c].map({True: "True", False: "False"})
```

จุดนี้คือ**bug ที่เจอจากการรันจริง**: 2 คอลัมน์นี้เป็นค่า True/False (boolean)
ในข้อมูลดิบ พอเราจะเอาไปทำเป็น "หมวดหมู่" (categorical) ให้ XGBoost อ่าน ปรากฏว่า
XGBoost ไม่ยอมรับหมวดหมู่ที่เป็น boolean (มันรองรับแค่ string หรือ integer)
`.map({True: "True", False: "False"})` เลยแปลง `True`/`False` (boolean) ให้กลาย
เป็น `"True"`/`"False"` (ตัวหนังสือ) แทน — ค่าที่หายไป (NaN) จะยังเป็น NaN เหมือนเดิม
เพราะ `.map()` จะปล่อยผ่านค่าที่หาไม่เจอใน dictionary

### `clean_geo()` — ความสะอาดเฉพาะข้อมูลพิกัด

```python
df["invalid_coords"] = invalid.astype(int)
df.loc[invalid, ["longitude", "latitude"]] = np.nan
```

แถวที่พิกัดผิดปกติ (จาก EDA ข้างบน) จะถูก**จดบันทึกไว้**เป็นคอลัมน์ตัวเลข 0/1
ชื่อ `invalid_coords` (เผื่อโมเดลอยากรู้ว่า "แถวนี้เคยมีปัญหาพิกัดนะ") แล้วค่อยตั้ง
พิกัดจริงเป็น `NaN` (ลบค่าที่ผิดทิ้ง เดี๋ยวจะไปซ่อมทีหลังในขั้นตอนที่ 4)

```python
for c in ["subvillage", "ward", "lga", "scheme_name", "funder", "installer"]:
    df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})
```

ทำความสะอาดตัวหนังสือ: `.str.strip()` ตัดช่องว่างหัวท้ายทิ้ง, `.str.lower()` แปลง
เป็นตัวพิมพ์เล็กหมด (กัน "Iringa" กับ "iringa" ถูกนับเป็นคนละค่ากัน) — ทำแบบนี้
เพราะคอลัมน์พวกนี้จะเอาไปนับความถี่ (frequency) ในขั้นตอนถัดไป ถ้าตัวพิมพ์ไม่
ตรงกันจะนับแยกกันผิด ๆ

---

## 4. สร้าง GIS Feature (บรรทัด 166-315) — ส่วนที่ซับซ้อนที่สุด

### ทำไมต้อง "project" พิกัด

```python
TZ_PROJECTED_CRS = CRS.from_proj4(
    "+proj=aeqd +lat_0=-6.5 +lon_0=35.0 +datum=WGS84 +units=m +no_defs"
)
```

พิกัด latitude/longitude ปกติ (เช่น -6.5, 35.0) วัดเป็น**องศา**บนทรงกลม ไม่ใช่
เส้นตรงบนกระดาษแบน ถ้าเอาไปคำนวณระยะทางตรง ๆ (เหมือนวัดเส้นตรงบนกระดาษ)
จะผิดพลาด เพราะเส้นละติจูด/ลองจิจูด 1 องศา แทนระยะทางจริงไม่เท่ากันในแต่ละจุด
ของโลก จึงต้อง **"project" (ฉาย)** พิกัดจากทรงกลมลงเป็นระนาบแบนหน่วยเมตรก่อน
ถึงจะคำนวณระยะทาง/จัดกลุ่มตามระยะทางได้แม่นยำ

โค้ดนี้เลือกใช้ **Azimuthal Equidistant (AEQD)** — ระบบที่ตั้งจุดศูนย์กลางไว้ที่
กึ่งกลางแทนซาเนีย (`lat_0=-6.5, lon_0=35.0`) แล้ววัดระยะทางจากจุดศูนย์กลางออกไป
ได้แม่นยำในทุกทิศทาง เหมาะกับงานระดับประเทศ (ถ้าใช้ระบบ UTM ซึ่งแบ่งโลกเป็น
โซนแคบ ๆ แทนซาเนียจะครอบคลุม 2 โซนพอดี ทำให้อีกฝั่งของประเทศคำนวณผิดเพี้ยน)

```python
def project_xy(lat, lon):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=WGS84)
    gdf_proj = gdf.to_crs(TZ_PROJECTED_CRS)
    return gdf_proj.geometry.x.values, gdf_proj.geometry.y.values
```

- `gpd.points_from_xy(lon, lat)` สร้าง "จุด" ทางภูมิศาสตร์จากตัวเลข lon/lat แต่ละคู่
- `gpd.GeoDataFrame(..., crs=WGS84)` บอกว่าจุดพวกนี้อยู่ในระบบพิกัดแบบ GPS ปกติ
  (`WGS84` คือชื่อระบบพิกัดมาตรฐานที่ GPS ทุกเครื่องใช้)
- `.to_crs(TZ_PROJECTED_CRS)` คือคำสั่ง **แปลงระบบพิกัด** จาก WGS84 (องศา) ไปเป็น
  AEQD (เมตร) ที่เราตั้งไว้ข้างบน
- `.geometry.x` / `.geometry.y` คือค่าพิกัดใหม่หลังแปลงแล้ว (หน่วยเป็นเมตรจากจุด
  ศูนย์กลาง)

ฟังก์ชันนี้ถูกเรียกใช้ทุกครั้งที่ต้องคำนวณระยะทางหรือจัดกลุ่มพื้นที่ในโค้ดส่วนถัดไป

### `validate_against_shapefile()` — ฟังก์ชันเสริม (ยังไม่ได้ใช้จริงในสคริปต์)

ฟังก์ชันนี้เตรียมไว้เผื่ออยากตรวจพิกัดให้แม่นกว่าการใช้กรอบสี่เหลี่ยมคร่าว ๆ
โดยเช็คว่าจุดนั้น "อยู่ในรูปร่างขอบเขตจริง" ของแทนซาเนียหรือไม่ (point-in-polygon)
— ต้องมีไฟล์ shapefile ขอบเขตจริงถึงจะใช้ได้ ถ้าไม่มีไฟล์จะข้ามไปเฉย ๆ

### `MAJOR_CITIES` + `haversine_km()` — คำนวณระยะทางถึงเมืองใหญ่

```python
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    ...
    return 2 * R * np.arcsin(np.sqrt(a))
```

Haversine formula คือสูตรคณิตศาสตร์มาตรฐานสำหรับคำนวณ "ระยะทางเส้นตรงบนผิว
ทรงกลม" ระหว่าง 2 จุดที่รู้ lat/lon (ต่างจาก `project_xy` ที่แปลงพิกัดทั้งชุดไปเป็น
ระนาบแบน — อันนี้คำนวณระยะทางระหว่างจุด 2 จุดโดยตรงโดยไม่ต้อง project ก่อน
ทั้งสองวิธีให้ผลใกล้เคียงกันในระยะไม่ไกลเกินไป) ใช้คำนวณว่าปั๊มแต่ละตัวอยู่ห่างจาก
เมืองใหญ่ที่ใกล้ที่สุดกี่กิโลเมตร (`dist_nearest_city_km`)

### `FreqEncoder` — คลาสสำหรับเข้ารหัสคอลัมน์ที่มีค่าไม่ซ้ำเยอะมาก

```python
class FreqEncoder(BaseEstimator, TransformerMixin):
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
```

`funder` (ผู้ให้ทุน) กับ `installer` (ผู้ติดตั้ง) มีชื่อไม่ซ้ำกันเป็นพัน ๆ ชื่อ ใส่เป็น
category ตรง ๆ จะทำให้โมเดลสับสน/ช้า วิธีแก้คือแปลงแต่ละชื่อเป็น **"ความถี่ที่ชื่อนี้
ปรากฏในข้อมูล"** แทน เช่นถ้า "Government" ปรากฏ 20% ของข้อมูลทั้งหมด ทุกแถวที่
`funder == "Government"` จะได้ค่า `funder_freq = 0.20`

- `fit()`: นับความถี่ของแต่ละชื่อ **จากข้อมูล train เท่านั้น** เก็บไว้ใน `self.maps_`
- `transform()`: เอาความถี่ที่นับไว้ไปแปะให้แต่ละแถว (`.map(m)`) ชื่อที่ไม่เคยเห็นตอน
  fit (เช่นชื่อใหม่ใน validation fold) จะได้ `NaN` แล้วถูกเติมเป็น `0.0` ด้วย `.fillna(0.0)`

**ทำไมต้องแยก fit กับ transform**: ถ้านับความถี่จากข้อมูล**ทั้งหมด**รวม
validation fold ด้วย จะเป็น data leakage (โมเดลได้ "แอบเห็น" ข้อมูลที่ควรใช้ทดสอบ)
คะแนนที่วัดได้จะดีเกินจริง

### `GeoFeaturizer` — หัวใจของการสร้าง feature เชิงพื้นที่

คลาสนี้ทำ 4 อย่าง: (1) ซ่อมพิกัดที่หายไป (2) จัดกลุ่มปั๊มตามตำแหน่ง (3) คำนวณ
ระยะทาง/ความหนาแน่น (4) นับความถี่ชื่อ

```python
def fit(self, X, y=None):
    self.region_coord_medians_ = (
        X.loc[X.invalid_coords == 0].groupby("region")[["latitude", "longitude"]].median()
    )
```

หา **ค่ากลาง (median) ของพิกัด แยกตามภูมิภาค** จากปั๊มที่พิกัดถูกต้องเท่านั้น
(`X.invalid_coords == 0` กรองแถวที่พิกัดดีออกมาก่อน) เตรียมไว้ใช้ "เดา" พิกัดของ
ปั๊มที่พิกัดหาย — ใช้ median ของภูมิภาคเดียวกัน สมเหตุสมผลกว่าการใช้ค่ากลางของ
ทั้งประเทศ เพราะแทนซาเนียกว้างมาก

```python
    xy = np.column_stack(project_xy(X_imputed.latitude.values, X_imputed.longitude.values))
    self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=RANDOM_STATE)
    self.kmeans_.fit(xy)
```

**KMeans** คืออัลกอริทึม "จัดกลุ่มจุดที่อยู่ใกล้กัน" โดยอัตโนมัติ — บอกว่าอยากได้กี่
กลุ่ม (`n_clusters=40`) แล้วมันจะหาตำแหน่งจุดศูนย์กลาง 40 จุดที่ทำให้ปั๊มแต่ละตัว
อยู่ใกล้ศูนย์กลางกลุ่มของตัวเองมากที่สุด ผลลัพธ์คือปั๊มแต่ละตัวจะได้ "หมายเลขกลุ่ม"
(0-39) ซึ่งกลายเป็น feature ชื่อ `spatial_cluster` — ปั๊มที่อยู่ในพื้นที่เดียวกันจะได้
หมายเลขเดียวกัน โมเดลใช้ตรงนี้จับ "ความเสี่ยงร่วมของพื้นที่" ได้ (ผลจาก feature
importance พบว่านี่คือ feature ที่สำคัญที่สุดในทั้งโมเดล!)

```python
    self.kdtree_ = cKDTree(xy)
```

สร้างโครงสร้างข้อมูล KD-tree จากตำแหน่งปั๊มทั้งหมด (หลัง project เป็นเมตรแล้ว)
เพื่อเอาไว้ค้นหา "เพื่อนบ้านใกล้เคียง" ได้เร็ว ใช้ในเมธอด `transform()`:

```python
def transform(self, X):
    ...
    X["spatial_cluster"] = self.kmeans_.predict(xy)
```
เอาโมเดล KMeans ที่ fit ไว้แล้ว มา "ทำนาย" ว่าจุดใหม่แต่ละจุดควรอยู่กลุ่มไหน

```python
    dists = [haversine_km(...) for lat_c, lon_c in MAJOR_CITIES.values()]
    X["dist_nearest_city_km"] = np.min(np.column_stack(dists), axis=1)
```
คำนวณระยะทางจากปั๊มแต่ละตัว **ไปยังเมืองใหญ่ทั้ง 15 เมือง** แล้วเลือกค่า**น้อย
ที่สุด** (`np.min(..., axis=1)`) = ระยะทางถึงเมืองที่ใกล้ที่สุด

```python
    radius_m = self.density_radius_km * 1000
    X["local_density"] = self.kdtree_.query_ball_point(xy, r=radius_m, return_length=True)
```
`query_ball_point` ถามว่า "รอบจุดนี้ในรัศมี 5 กม. (`radius_m`) มีปั๊มอื่นกี่ตัว"
(`return_length=True` ให้คืนแค่**จำนวน**ไม่ต้องคืนรายชื่อจุด) ยิ่งค่าเยอะแปลว่าพื้นที่
นั้นมีปั๊มหนาแน่น

```python
    nn_dist, _ = self.kdtree_.query(xy, k=2)
    X["nearest_pump_m"] = np.where(nn_dist[:, 0] < 1e-6, nn_dist[:, 1], nn_dist[:, 0])
```
`.query(xy, k=2)` หา "2 จุดที่ใกล้ที่สุด" ของแต่ละจุด — ใส่ `k=2` (ไม่ใช่ 1) เพราะ
จุดที่ใกล้ตัวเองที่สุดคือ**ตัวมันเอง**เสมอ (ระยะทาง 0) เราต้องการเพื่อนบ้าน**อันดับ
สอง** จริง ๆ บรรทัด `np.where` เช็คว่าถ้าระยะทางอันดับ 1 ใกล้ 0 มาก (คือเจอตัวเอง)
ให้ใช้ค่าอันดับ 2 แทน

---

## 5. เลือกชุด Feature ที่จะใช้เข้าโมเดล (บรรทัด 322-351)

```python
BASELINE_NUM = ["amount_tsh", "gps_height", "population", "age",
                 "funder_freq", "installer_freq"]
BASELINE_CAT = ["basin", "region", "public_meeting", ...]
```

`BASELINE_*` คือชุด 18 feature ที่ repo ต้นฉบับใช้ (ไม่มีข้อมูลตำแหน่งเลย)
`GIS_NUM`/`GIS_CAT` คือ baseline **บวก**คอลัมน์ตำแหน่งทั้งหมดที่สร้างไว้ในขั้นตอน 4
(latitude, longitude, spatial_cluster, ระยะทาง, ความหนาแน่น ฯลฯ) — แยก 2 ชุดนี้
ไว้เพื่อจะได้เทรน 2 โมเดลมาเทียบกันตรง ๆ ว่า "มีข้อมูลตำแหน่งช่วยจริงมั้ย"

```python
def fit_category_levels(df, cat_cols):
    return {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols}
```
จำ **"รายการค่าที่เป็นไปได้"** ของแต่ละคอลัมน์หมวดหมู่ไว้ (เช่น `region` มีค่าที่
เป็นไปได้ 21 ค่า) — สำคัญเพราะถ้าไม่จำไว้ ตอนแบ่งข้อมูลเป็น train/validation ใน
แต่ละรอบ CV บางค่าอาจไม่ปรากฏใน validation fold ทำให้ตัวเลขรหัสภายใน (category
code) เพี้ยนไม่ตรงกับตอน fit — จำไว้แล้วบังคับใช้รายการเดียวกันเสมอจะปลอดภัยกว่า

```python
def to_model_frame(df, num_cols, cat_cols, cat_levels):
    X = df[num_cols + cat_cols].copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X
```
ฟังก์ชันนี้คือ "ขั้นตอนสุดท้ายก่อนเข้าโมเดล": เลือกเฉพาะคอลัมน์ที่ต้องการ, แปลง
คอลัมน์ตัวเลขให้เป็นตัวเลขจริง ๆ (`errors="coerce"` = ถ้าแปลงไม่ได้ให้เป็น `NaN`
แทนที่จะ error), แปลงคอลัมน์หมวดหมู่เป็น pandas category type โดยบังคับใช้
รายการค่าที่จำไว้จาก `fit_category_levels` เท่านั้น

---

## 6. เทรนโมเดล + วัดผล (บรรทัด 358-514)

### F2-score คืออะไร ทำไมไม่ใช้ accuracy เฉย ๆ

```python
def f2_macro(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average="macro")
```
งานบำรุงรักษาเชิงป้องกัน การ**พลาดปั๊มที่กำลังจะเสีย**(false negative) อันตราย
กว่าการ**ส่งช่างไปดูปั๊มที่จริง ๆ ยังดีอยู่**(false positive) — F-beta score คือค่าที่
ถ่วงน้ำหนักระหว่าง precision (แม่นแค่ไหนตอนทายว่า "เสีย") กับ recall (จับปั๊มที่
เสียจริงได้ครบแค่ไหน) `beta=2` แปลว่าให้ความสำคัญกับ recall มากกว่า precision
2 เท่า `average="macro"` แปลว่าคำนวณแยกแต่ละ class (3 class) แล้วเอามาเฉลี่ย
เท่า ๆ กัน (ไม่ให้ class ที่มีข้อมูลเยอะกว่ามีอิทธิพลมากกว่า)

### เข้ารหัส label (บั๊กที่เจอจากการรันจริง)

```python
def encode_labels(y):
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    return y_enc, le
```
XGBoost (ตั้งแต่เวอร์ชัน 1.3.2) ต้องการคำตอบ (`y`) เป็น**ตัวเลข** 0, 1, 2 เท่านั้น
ไม่รับ string อย่าง `"functional"` ตรง ๆ (ต่างจาก sklearn ที่รับ string ได้เลย)
`LabelEncoder` คือเครื่องมือแปลงชื่อ class เป็นตัวเลข: `fit_transform(y)` เรียนรู้ว่า
ชื่อไหนควรแทนด้วยเลขอะไร แล้วแปลงให้เลย ทีหลังใช้ `le.inverse_transform()`
แปลงตัวเลขกลับเป็นชื่อเดิมตอนจะดูรายงานผล (ให้อ่านง่าย)

### `make_xgb()` — ตั้งค่าโมเดล XGBoost

```python
return XGBClassifier(
    n_estimators=300,      # จำนวน "ต้นไม้ตัดสินใจ" ที่จะสร้างต่อ ๆ กัน
    learning_rate=0.08,    # แต่ละต้นเรียนรู้จากความผิดพลาดของต้นก่อนหน้าแค่ไหน (ค่าน้อย = เรียนช้าแต่มั่นคง)
    max_depth=6,           # แต่ละต้นลึกได้แค่ไหน (ลึกมาก = จำรายละเอียดเกินไป/overfit)
    tree_method="hist",    # วิธีสร้างต้นไม้แบบเร็ว (แบ่งข้อมูลเป็น histogram ก่อนหาจุดตัด)
    enable_categorical=True,  # อนุญาตให้ป้อนคอลัมน์ category dtype ตรง ๆ ไม่ต้อง one-hot เอง
    eval_metric="mlogloss",   # ตัวชี้วัดภายในที่โมเดลใช้เช็คตัวเองระหว่างเทรน
    random_state=RANDOM_STATE,  # เลขสุ่มคงที่ ให้ผลรันซ้ำได้เหมือนเดิมทุกครั้ง
    n_jobs=-1,              # ใช้ CPU ทุก core ที่มีช่วยคำนวณ
)
```

โมเดล XGBoost ทำงานแบบ "gradient boosting": สร้างต้นไม้ตัดสินใจ (decision tree)
ทีละต้น ต้นแรกทายมั่ว ๆ ต้นถัดไปพยายาม**แก้ไขความผิดพลาด**ของต้นก่อนหน้า ทำ
แบบนี้ซ้ำ 300 รอบ (`n_estimators=300`) แล้วเอาผลโหวตจากทุกต้นมารวมกันเป็นคำตอบ
สุดท้าย

### `GeoFrameBuilder` / `BaselineFrameBuilder` — ห่อทุกอย่างเป็นชิ้นเดียวที่ใช้กับ sklearn ได้

```python
class GeoFrameBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.geo_.fit(X, y)
        X_transformed = self.geo_.transform(X)
        self.cat_levels_ = fit_category_levels(X_transformed, GIS_CAT)
        return self

    def transform(self, X):
        X2 = self.geo_.transform(X)
        return to_model_frame(X2, GIS_NUM, GIS_CAT, self.cat_levels_)
```

คลาสนี้แค่ **เอาขั้นตอนที่อธิบายไปแล้วทั้งหมด (GeoFeaturizer + to_model_frame)
มาเรียงต่อกันเป็นขั้นตอนเดียว** ที่มี `.fit()`/`.transform()` ตามมาตรฐาน sklearn
เหตุผลที่ต้องทำแบบนี้: จะได้เอาไปใส่ใน `Pipeline([...])` แล้วให้ `cross_val_score`
จัดการเรียก `.fit()` เฉพาะกับ training fold ของแต่ละรอบ CV ให้เองอัตโนมัติ — ไม่
ต้องเขียน loop 5 รอบเอง ซึ่งเสี่ยงเขียนผิดแล้วเกิด leakage สูงกว่ามาก

### `evaluate()` — รันเปรียบเทียบโมเดลด้วย cross-validation

```python
def evaluate(df_clean, label, builder, y):
    pipe = Pipeline([("features", builder), ("model", make_xgb())])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, df_clean, y, cv=skf, scoring=F2_SCORER, n_jobs=1)
    ...
```

- `Pipeline([("features", builder), ("model", make_xgb())])` มัดขั้นตอน "แปลง
  ข้อมูล" กับ "เทรนโมเดล" ให้เป็นก้อนเดียว
- `StratifiedKFold(n_splits=5, ...)` คือวิธีแบ่งข้อมูลเป็น 5 ส่วนสำหรับ CV — "Stratified"
  แปลว่าพยายามให้สัดส่วนของแต่ละ class (functional/needs repair/non functional)
  ใกล้เคียงกันในทุกส่วน ไม่ให้บางส่วนมีแต่ class เดียวโดยบังเอิญ
- `cross_val_score(...)` คือคำสั่งเดียวที่ทำทั้งหมด: แบ่งข้อมูล 5 ส่วน → เทรนด้วย
  4 ส่วน → ทดสอบกับ 1 ส่วนที่เหลือ → วนซ้ำ 5 รอบ (สลับว่าส่วนไหนเป็นส่วนทดสอบ)
  → คืนคะแนนทั้ง 5 รอบมาเป็น array

### `main()` — เมื่อรันไฟล์นี้ จะเกิดอะไรขึ้นบ้าง ตามลำดับ

1. โหลดข้อมูล + ทำ EDA (นับปัญหา, เขียนรายงาน)
2. ทำความสะอาดข้อมูล (`clean_common`, `clean_geo`)
3. เข้ารหัส label เป็นตัวเลข (`encode_labels`)
4. เทรน+วัดผลโมเดล **baseline** (ไม่มี geo) ด้วย CV 5 fold
5. เทรน+วัดผลโมเดล **gis_enhanced** (มี geo ครบ) ด้วย CV 5 fold
6. บันทึกตารางเทียบคะแนนทั้ง 2 โมเดล เป็น CSV + กราฟแท่ง
7. เทรนโมเดล gis_enhanced อีกรอบแบบ `cross_val_predict` (ได้คำทำนายของ**ทุกแถว**
   จากตอนที่แถวนั้นอยู่ใน validation fold) เอาไปทำ classification report +
   confusion matrix — สะท้อนภาพว่าโมเดลทายผิดแบบไหนบ้าง
8. เทรนโมเดลสุดท้ายด้วยข้อมูล**ทั้งหมด** (ไม่แบ่ง fold แล้ว) เพื่อดู feature
   importance ว่า feature ไหนสำคัญที่สุด
9. บันทึกข้อมูลที่ clean + สร้าง feature ครบแล้ว เป็นไฟล์ CSV ไว้ใช้ต่อ

ทุกไฟล์ผลลัพธ์ (`.csv`, `.png`, `.txt`) จะถูกเซฟไว้ในโฟลเดอร์ `OUT_DIR` ที่ตั้งไว้
บนสุดของไฟล์

---

## สรุปภาพรวมอีกที

ถ้าให้สรุปเป็นประโยคเดียว: โค้ดนี้**ทำความสะอาดข้อมูลปั๊มน้ำ → เปลี่ยนพิกัด
ดิบให้กลายเป็น feature ที่มีความหมาย (กลุ่มพื้นที่, ระยะทาง, ความหนาแน่น) →
เอาไปเทรนโมเดล XGBoost 2 แบบเปรียบเทียบกันด้วยวิธีที่ป้องกัน data leakage
อย่างระมัดระวังทุกขั้นตอน** เพื่อพิสูจน์ว่าข้อมูลตำแหน่งช่วยทำนายว่าปั๊มจะเสียได้
ดีขึ้นจริงหรือไม่
